"""
Live odds streaming engine.

Polls one or more odds sources on a configurable interval and emits
change events to registered callbacks.

Design
------
- Thread-safe: OddsStream runs in a daemon thread; callbacks are called
  from that thread so they must be non-blocking (or queue work).
- Exponential back-off on consecutive failures (cap: 60 s).
- Change detection: a snapshot is only emitted when implied probabilities
  shift by at least `min_delta` for any outcome, avoiding spam on tiny
  rounding noise from the bookmaker API.
- Steam detection: rapid implied-prob moves (>4 pp within 5 min) trigger
  a dedicated `on_steam` callback alongside the normal `on_update`.
- Health monitor: exposes `healthy`, `last_success_ts`, `consecutive_failures`.
- Graceful stop: call `stop()` or use as context manager.

Usage
-----
    def handle_update(snapshot: OddsSnapshot):
        print(snapshot)

    stream = OddsStream(
        sources={"pinnacle": fetch_pinnacle_odds},
        interval=300,
        on_update=handle_update,
    )
    stream.start()
    ...
    stream.stop()
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

_STEAM_WINDOW_SECS = 300.0  # 5 minutes
_STEAM_THRESHOLD = 0.04  # 4 pp implied-prob move


@dataclass
class OddsSnapshot:
    """A single odds observation from one source for one fixture."""

    fixture_id: str
    source: str
    timestamp: float  # Unix epoch seconds
    home_odds: float
    draw_odds: float
    away_odds: float
    # Optional extended markets
    over_25_odds: Optional[float] = None
    under_25_odds: Optional[float] = None
    btts_yes_odds: Optional[float] = None

    # ------------------------------------------------------------------
    def implied(self) -> Dict[str, float]:
        """Vig-free implied probabilities for 1×2 market."""
        ih = 1.0 / self.home_odds if self.home_odds > 0 else 0.0
        id_ = 1.0 / self.draw_odds if self.draw_odds > 0 else 0.0
        ia = 1.0 / self.away_odds if self.away_odds > 0 else 0.0
        total = ih + id_ + ia or 1.0
        return {"home": ih / total, "draw": id_ / total, "away": ia / total}

    def overround(self) -> float:
        ih = 1.0 / self.home_odds if self.home_odds > 0 else 0.0
        id_ = 1.0 / self.draw_odds if self.draw_odds > 0 else 0.0
        ia = 1.0 / self.away_odds if self.away_odds > 0 else 0.0
        return ih + id_ + ia - 1.0


@dataclass
class StreamHealth:
    """Runtime health metrics for the streaming engine."""

    source: str
    consecutive_failures: int = 0
    total_polls: int = 0
    total_changes: int = 0
    total_errors: int = 0
    last_success_ts: float = 0.0
    last_error_msg: str = ""

    @property
    def healthy(self) -> bool:
        return self.consecutive_failures < 5

    @property
    def uptime_pct(self) -> float:
        if self.total_polls == 0:
            return 0.0
        return round((self.total_polls - self.total_errors) / self.total_polls * 100, 1)


# ---------------------------------------------------------------------------
# Change detector
# ---------------------------------------------------------------------------


class _ChangeDetector:
    """
    Tracks the last emitted snapshot per fixture and source.
    Emits True only when implied probabilities shift beyond `min_delta`.
    """

    def __init__(self, min_delta: float = 0.005) -> None:
        self._last: Dict[str, Dict[str, float]] = {}
        self._min_delta = min_delta

    def has_changed(self, snap: OddsSnapshot) -> bool:
        key = f"{snap.fixture_id}@{snap.source}"
        curr = snap.implied()
        if key not in self._last:
            self._last[key] = curr
            return True
        prev = self._last[key]
        if any(abs(curr[k] - prev[k]) >= self._min_delta for k in curr):
            self._last[key] = curr
            return True
        return False

    def reset(self, fixture_id: str, source: str) -> None:
        self._last.pop(f"{fixture_id}@{source}", None)


# ---------------------------------------------------------------------------
# Steam detector (per fixture, across sources)
# ---------------------------------------------------------------------------


class _SteamDetector:
    """
    Detects rapid implied-probability moves (steam) per fixture.

    Keeps a rolling deque of the last N snapshots per fixture and checks
    whether any outcome shifted >STEAM_THRESHOLD within STEAM_WINDOW_SECS.
    """

    def __init__(
        self,
        window_secs: float = _STEAM_WINDOW_SECS,
        threshold: float = _STEAM_THRESHOLD,
        history_len: int = 20,
    ) -> None:
        self._window = window_secs
        self._threshold = threshold
        # fixture_id → deque[OddsSnapshot]
        self._history: Dict[str, Deque[OddsSnapshot]] = {}
        self._history_len = history_len

    def record(self, snap: OddsSnapshot) -> Optional[tuple[str, float]]:
        """
        Record a snapshot and check for steam.

        Returns (market, magnitude) if steam detected, else None.
        """
        fid = snap.fixture_id
        if fid not in self._history:
            self._history[fid] = deque(maxlen=self._history_len)

        dq = self._history[fid]
        dq.append(snap)

        # Remove entries outside the time window
        cutoff = snap.timestamp - self._window
        while dq and dq[0].timestamp < cutoff:
            dq.popleft()

        if len(dq) < 2:
            return None

        curr_imp = snap.implied()
        # Compare against oldest in-window snapshot
        oldest_imp = dq[0].implied()

        best_market: str = ""
        best_mag: float = 0.0

        for mkt in ("home", "draw", "away"):
            shift = curr_imp[mkt] - oldest_imp[mkt]
            if shift >= self._threshold and shift > best_mag:
                best_mag = shift
                best_market = mkt

        return (best_market, round(best_mag, 4)) if best_market else None


# ---------------------------------------------------------------------------
# Source wrapper
# ---------------------------------------------------------------------------

# A source callable must accept no arguments and return a list of dicts,
# each dict with keys: fixture_id, home_odds, draw_odds, away_odds,
# and optionally: over_25_odds, under_25_odds, btts_yes_odds.
SourceFn = Callable[[], List[Dict]]


def _safe_float(v) -> Optional[float]:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _parse_snapshot(raw: dict, source: str, timestamp: float) -> Optional[OddsSnapshot]:
    """Convert a raw dict to OddsSnapshot, returning None on invalid data."""
    try:
        home_odds = float(raw["home_odds"])
        draw_odds = float(raw["draw_odds"])
        away_odds = float(raw["away_odds"])
        if home_odds <= 1.0 or draw_odds <= 1.0 or away_odds <= 1.0:
            return None
        return OddsSnapshot(
            fixture_id=str(raw["fixture_id"]),
            source=source,
            timestamp=timestamp,
            home_odds=home_odds,
            draw_odds=draw_odds,
            away_odds=away_odds,
            over_25_odds=_safe_float(raw.get("over_25_odds")),
            under_25_odds=_safe_float(raw.get("under_25_odds")),
            btts_yes_odds=_safe_float(raw.get("btts_yes_odds")),
        )
    except (KeyError, TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Main streaming engine
# ---------------------------------------------------------------------------


class OddsStream:
    """
    Multi-source live odds polling engine.

    Parameters
    ----------
    sources      : mapping of source_name → callable() → list[dict]
    interval     : polling interval in seconds (default 300 = 5 min)
    min_delta    : minimum implied-prob shift to trigger on_update (default 0.005)
    on_update    : callback(snapshot: OddsSnapshot)
    on_steam     : callback(snapshot: OddsSnapshot, market: str, magnitude: float)
    on_error     : callback(source: str, exc: Exception)
    on_cycle_end : callback(health: dict[source, StreamHealth])
    max_backoff  : cap for exponential back-off in seconds (default 60)
    """

    def __init__(
        self,
        sources: Dict[str, SourceFn],
        interval: float = 300.0,
        min_delta: float = 0.005,
        on_update: Optional[Callable[[OddsSnapshot], None]] = None,
        on_steam: Optional[Callable[[OddsSnapshot, str, float], None]] = None,
        on_error: Optional[Callable[[str, Exception], None]] = None,
        on_cycle_end: Optional[Callable[[Dict[str, "StreamHealth"]], None]] = None,
        max_backoff: float = 60.0,
    ) -> None:
        if not sources:
            raise ValueError("At least one odds source must be provided.")

        self._sources = sources
        self._interval = interval
        self._min_delta = min_delta
        self._on_update = on_update
        self._on_steam = on_steam
        self._on_error = on_error
        self._on_cycle_end = on_cycle_end
        self._max_backoff = max_backoff

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        self._health: Dict[str, StreamHealth] = {
            name: StreamHealth(source=name) for name in sources
        }
        self._change_detectors: Dict[str, _ChangeDetector] = {
            name: _ChangeDetector(min_delta) for name in sources
        }
        self._steam_detector = _SteamDetector()

        # Per-source back-off state
        self._backoff: Dict[str, float] = {name: 1.0 for name in sources}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> "OddsStream":
        """Start the streaming loop in a background daemon thread."""
        if self._thread and self._thread.is_alive():
            return self
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._loop, name="OddsStream", daemon=True
        )
        self._thread.start()
        logger.info(
            "OddsStream started (interval=%.0fs, sources=%s)",
            self._interval,
            list(self._sources),
        )
        return self

    def stop(self, timeout: float = 10.0) -> None:
        """Signal the streaming loop to stop and wait for thread exit."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=timeout)
        logger.info("OddsStream stopped.")

    def health(self) -> Dict[str, StreamHealth]:
        """Return a copy of all source health objects."""
        with self._lock:
            return dict(self._health)

    def is_running(self) -> bool:
        return bool(
            self._thread and self._thread.is_alive() and not self._stop_event.is_set()
        )

    # Context manager support
    def __enter__(self) -> "OddsStream":
        return self.start()

    def __exit__(self, *_) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Internal loop
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            cycle_start = time.monotonic()

            for source_name, fetch_fn in self._sources.items():
                if self._stop_event.is_set():
                    break
                self._poll_source(source_name, fetch_fn)

            with self._lock:
                health_copy = dict(self._health)

            if self._on_cycle_end:
                try:
                    self._on_cycle_end(health_copy)
                except Exception:
                    pass

            # Sleep the remainder of the interval
            elapsed = time.monotonic() - cycle_start
            remaining = max(0.0, self._interval - elapsed)
            self._stop_event.wait(timeout=remaining)

    def _poll_source(self, name: str, fetch_fn: SourceFn) -> None:
        """Poll one source, handle errors with exponential back-off."""
        health = self._health[name]
        detector = self._change_detectors[name]
        health.total_polls += 1

        try:
            raw_list = fetch_fn()
            ts = time.time()

            if not isinstance(raw_list, list):
                raise TypeError(
                    f"Source '{name}' must return list[dict], got {type(raw_list)}"
                )

            changed_count = 0
            for raw in raw_list:
                snap = _parse_snapshot(raw, name, ts)
                if snap is None:
                    continue

                if detector.has_changed(snap):
                    changed_count += 1
                    self._dispatch_update(snap)

                    steam = self._steam_detector.record(snap)
                    if steam:
                        market, magnitude = steam
                        self._dispatch_steam(snap, market, magnitude)

            health.last_success_ts = ts
            health.consecutive_failures = 0
            health.total_changes += changed_count
            self._backoff[name] = 1.0  # reset back-off on success

            logger.debug(
                "OddsStream [%s] polled %d fixtures, %d changed.",
                name,
                len(raw_list),
                changed_count,
            )

        except Exception as exc:
            health.consecutive_failures += 1
            health.total_errors += 1
            health.last_error_msg = str(exc)

            backoff = min(self._backoff[name] * 2, self._max_backoff)
            self._backoff[name] = backoff

            logger.warning(
                "OddsStream [%s] error (failure #%d, backoff=%.0fs): %s",
                name,
                health.consecutive_failures,
                backoff,
                exc,
            )

            if self._on_error:
                try:
                    self._on_error(name, exc)
                except Exception:
                    pass

            # Brief back-off sleep (interruptible)
            self._stop_event.wait(timeout=backoff)

    def _dispatch_update(self, snap: OddsSnapshot) -> None:
        if self._on_update:
            try:
                self._on_update(snap)
            except Exception as exc:
                logger.warning("on_update callback raised: %s", exc)

    def _dispatch_steam(self, snap: OddsSnapshot, market: str, mag: float) -> None:
        logger.info(
            "STEAM [%s] fixture=%s market=%s magnitude=%.4f",
            snap.source,
            snap.fixture_id,
            market,
            mag,
        )
        if self._on_steam:
            try:
                self._on_steam(snap, market, mag)
            except Exception as exc:
                logger.warning("on_steam callback raised: %s", exc)


# ---------------------------------------------------------------------------
# Convenience: build a stream from a simple polling function
# ---------------------------------------------------------------------------


def build_stream_from_collector(
    fetch_fn: SourceFn,
    source: str = "default",
    interval: float = 300.0,
    on_update: Optional[Callable[[OddsSnapshot], None]] = None,
    on_steam: Optional[Callable[[OddsSnapshot, str, float], None]] = None,
) -> OddsStream:
    """
    Shortcut to create an OddsStream from a single fetcher function.

    Parameters
    ----------
    fetch_fn  : callable() → list[dict]  (each dict has fixture_id, home_odds, …)
    source    : display name for this source
    interval  : polling cadence in seconds
    on_update : callback invoked when odds change
    on_steam  : callback invoked when a steam move is detected
    """
    return OddsStream(
        sources={source: fetch_fn},
        interval=interval,
        on_update=on_update,
        on_steam=on_steam,
    )

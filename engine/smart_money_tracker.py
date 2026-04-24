"""
smart_money_tracker.py
======================
Monitors Betfair Exchange market data for volume anomalies that indicate
professional ("smart money") activity.  A volume spike combined with an odds
movement signals that a syndicate has placed a large bet.  The module emits
SmartMoneyAlert notifications via an optional callback.

Two layers:
  BetfairExchangeClient  — HTTP client for Betfair Exchange REST API (APING).
  SmartMoneyTracker      — stateful anomaly detector.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BETFAIR_API_ENDPOINT = "https://api.betfair.com/exchange/betting/rest/v1.0/"
_BETFAIR_LOGIN_ENDPOINT = "https://identitysso-cert.betfair.com/api/certlogin"
_DEFAULT_WINDOW_SECONDS = 300  # 5-minute baseline window
_VOLUME_SPIKE_MULTIPLIER = 3.0  # alert if volume > 3× baseline
_MIN_SPIKE_VOLUME = 5000.0  # minimum £ to trigger (ignore noise)
_ODDS_MOVE_THRESHOLD = 0.05  # minimum odds move (abs) to confirm smart money

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class MarketSnapshot:
    market_id: str  # Betfair market ID e.g. "1.234567890"
    selection_id: int  # runner ID
    selection_name: str
    timestamp: float  # unix epoch
    best_back: float  # best back price (lay perspective: price you can back at)
    best_lay: float  # best lay price
    matched_volume: float  # total matched £ on this selection
    available_to_back: float  # liquidity at best back


@dataclass
class SmartMoneyAlert:
    market_id: str
    selection_id: int
    selection_name: str
    timestamp: float
    volume_5min: float  # matched in last 5 minutes
    volume_baseline: float  # typical 5-min volume
    volume_ratio: float  # volume_5min / volume_baseline
    odds_before: float  # price at start of spike window
    odds_after: float  # current price
    odds_move: float  # odds_before - odds_after (negative = shortening = money on)
    direction: str  # "BACK" if odds shortened (money backing), "LAY" otherwise
    confidence: str  # "HIGH" | "MEDIUM" | "LOW"

    def __str__(self) -> str:
        return (
            f"SMART MONEY [{self.selection_name}] "
            f"vol={self.volume_5min:,.0f}£ ({self.volume_ratio:.1f}× baseline) "
            f"odds {self.odds_before:.2f}→{self.odds_after:.2f} "
            f"dir={self.direction} conf={self.confidence}"
        )


# ---------------------------------------------------------------------------
# BetfairExchangeClient
# ---------------------------------------------------------------------------


class BetfairExchangeClient:
    """
    Thin wrapper around Betfair Exchange APING REST API.

    Authentication: session token obtained via cert-based login or stored
    app key + session.  The caller provides app_key and session_token directly
    (obtained externally).
    """

    def __init__(self, app_key: str, session_token: str, timeout: int = 10) -> None:
        self._app_key = app_key
        self._session_token = session_token
        self._timeout = timeout

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _request(self, operation: str, params: dict) -> dict:
        """
        POST to APING endpoint with JSON body.
        Headers: X-Application: app_key, X-Authentication: session_token,
                 Content-Type: application/json
        """
        url = _BETFAIR_API_ENDPOINT + operation + "/"
        payload = json.dumps(params).encode("utf-8")
        headers = {
            "X-Application": self._app_key,
            "X-Authentication": self._session_token,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
        logger.debug("APING %s params=%s", operation, params)
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                raw = resp.read()
                data = json.loads(raw.decode("utf-8"))
                logger.debug("APING %s response size=%d bytes", operation, len(raw))
                return data
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            logger.error("APING %s HTTP %s: %s", operation, exc.code, body[:400])
            raise
        except urllib.error.URLError as exc:
            logger.error("APING %s network error: %s", operation, exc.reason)
            raise
        except json.JSONDecodeError as exc:
            logger.error("APING %s invalid JSON: %s", operation, exc)
            raise

    # ------------------------------------------------------------------
    # Public API methods
    # ------------------------------------------------------------------

    def list_market_catalogue(
        self,
        event_type_ids: List[str],
        market_countries: List[str],
        max_results: int = 100,
    ) -> List[dict]:
        """
        Return list of markets from listMarketCatalogue.
        filter: {"eventTypeIds": event_type_ids,
                 "marketCountries": market_countries,
                 "marketTypeCodes": ["MATCH_ODDS"]}
        marketProjection: ["RUNNER_DESCRIPTION", "EVENT"]
        """
        params = {
            "filter": {
                "eventTypeIds": event_type_ids,
                "marketCountries": market_countries,
                "marketTypeCodes": ["MATCH_ODDS"],
            },
            "marketProjection": ["RUNNER_DESCRIPTION", "EVENT"],
            "maxResults": max_results,
        }
        result = self._request("listMarketCatalogue", params)
        # APING returns a list directly for listMarketCatalogue
        if isinstance(result, list):
            logger.info("listMarketCatalogue returned %d markets", len(result))
            return result
        # Some error responses wrap in a dict
        logger.warning("listMarketCatalogue unexpected response type: %s", type(result))
        return []

    def list_market_book(self, market_ids: List[str]) -> List[dict]:
        """
        Return market book data (prices + volumes) from listMarketBook.
        priceProjection: {"priceData": ["EX_BEST_OFFERS", "EX_TRADED"]}
        orderProjection: "EXECUTABLE"
        matchProjection: "NO_ROLLUP"
        """
        params = {
            "marketIds": market_ids,
            "priceProjection": {
                "priceData": ["EX_BEST_OFFERS", "EX_TRADED"],
            },
            "orderProjection": "EXECUTABLE",
            "matchProjection": "NO_ROLLUP",
        }
        result = self._request("listMarketBook", params)
        if isinstance(result, list):
            logger.info(
                "listMarketBook returned %d market books for %d ids",
                len(result),
                len(market_ids),
            )
            return result
        logger.warning("listMarketBook unexpected response type: %s", type(result))
        return []

    def extract_snapshots(self, market_books: List[dict]) -> List[MarketSnapshot]:
        """
        Parse listMarketBook response into list of MarketSnapshot.
        For each runner:
          best back = runners[].ex.availableToBack[0].price
          matched volume = runners[].totalMatched
        """
        snapshots: List[MarketSnapshot] = []
        now = time.time()

        for book in market_books:
            market_id: str = book.get("marketId", "")
            runners: List[dict] = book.get("runners", [])
            for runner in runners:
                selection_id: int = runner.get("selectionId", 0)
                selection_name: str = runner.get("runnerName", str(selection_id))
                total_matched: float = float(runner.get("totalMatched", 0.0))

                ex: dict = runner.get("ex", {})
                atb: List[dict] = ex.get("availableToBack", [])
                atl: List[dict] = ex.get("availableToLay", [])

                best_back: float = float(atb[0].get("price", 0.0)) if atb else 0.0
                best_lay: float = float(atl[0].get("price", 0.0)) if atl else 0.0
                avail_to_back: float = float(atb[0].get("size", 0.0)) if atb else 0.0

                snap = MarketSnapshot(
                    market_id=market_id,
                    selection_id=selection_id,
                    selection_name=selection_name,
                    timestamp=now,
                    best_back=best_back,
                    best_lay=best_lay,
                    matched_volume=total_matched,
                    available_to_back=avail_to_back,
                )
                snapshots.append(snap)
                logger.debug(
                    "Snapshot %s/%d vol=%.2f back=%.3f lay=%.3f",
                    market_id,
                    selection_id,
                    total_matched,
                    best_back,
                    best_lay,
                )

        return snapshots


# ---------------------------------------------------------------------------
# SmartMoneyTracker
# ---------------------------------------------------------------------------


class SmartMoneyTracker:
    """
    Detects smart-money volume spikes on Betfair Exchange.

    Call update(snapshot) each time new market data arrives.
    Fires on_alert callback when anomaly detected.

    Parameters
    ----------
    window_seconds       : rolling window for baseline (default 300s = 5 min)
    spike_multiplier     : volume ratio to trigger alert (default 3.0×)
    min_spike_volume     : minimum £ in window to trigger (default £5,000)
    odds_move_threshold  : minimum odds movement to confirm (default 0.05)
    on_alert             : optional callback(SmartMoneyAlert)
    """

    def __init__(
        self,
        window_seconds: float = _DEFAULT_WINDOW_SECONDS,
        spike_multiplier: float = _VOLUME_SPIKE_MULTIPLIER,
        min_spike_volume: float = _MIN_SPIKE_VOLUME,
        odds_move_threshold: float = _ODDS_MOVE_THRESHOLD,
        on_alert: Optional[Callable[[SmartMoneyAlert], None]] = None,
    ) -> None:
        self._window = window_seconds
        self._spike_mult = spike_multiplier
        self._min_vol = min_spike_volume
        self._odds_move_thresh = odds_move_threshold
        self._on_alert = on_alert
        # (market_id, selection_id) → deque of (timestamp, matched_volume, best_back)
        self._history: Dict[Tuple[str, int], Deque[Tuple[float, float, float]]] = {}
        self._alerts: List[SmartMoneyAlert] = []

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def update(self, snapshot: MarketSnapshot) -> Optional[SmartMoneyAlert]:
        """
        Feed a new market snapshot.  Returns SmartMoneyAlert if spike detected,
        else None.

        Algorithm
        ---------
        1. Append (timestamp, matched_volume, best_back) to history deque.
        2. Trim entries older than window_seconds.
        3. Compute volume traded in window = latest_vol - oldest_vol
           (matched_volume is cumulative on Betfair).
        4. Compute baseline = average volume across the last 5 prior windows
           (we synthesise these from history older than the current window).
        5. If volume_window > spike_multiplier × baseline AND
              volume_window > min_spike_volume:
             Check odds move: odds_start (oldest entry in window) − odds_now.
             If abs(odds_move) >= odds_move_threshold → emit alert.
        """
        key: Tuple[str, int] = (snapshot.market_id, snapshot.selection_id)
        if key not in self._history:
            self._history[key] = deque()

        hist = self._history[key]
        entry: Tuple[float, float, float] = (
            snapshot.timestamp,
            snapshot.matched_volume,
            snapshot.best_back,
        )
        hist.append(entry)

        # ------------------------------------------------------------------
        # Trim entries that fall outside the *baseline collection* horizon.
        # We need up to 5 prior windows + 1 current window of history so that
        # we can compute a meaningful baseline.  Keep at most 6 × window worth
        # of data; data beyond that can be discarded.
        # ------------------------------------------------------------------
        horizon = snapshot.timestamp - 6 * self._window
        while hist and hist[0][0] < horizon:
            hist.popleft()

        # Need at least 2 data points to compute anything
        if len(hist) < 2:
            logger.debug(
                "update %s/%d: insufficient history (%d pts)",
                snapshot.market_id,
                snapshot.selection_id,
                len(hist),
            )
            return None

        now_ts = snapshot.timestamp
        window_start_ts = now_ts - self._window

        # ------------------------------------------------------------------
        # Identify entries inside the current spike window vs. before it.
        # ------------------------------------------------------------------
        # Entries in the current window
        window_entries = [e for e in hist if e[0] >= window_start_ts]
        # Entries before the current window (used for baseline)
        pre_window_entries = [e for e in hist if e[0] < window_start_ts]

        if not window_entries:
            return None

        # Volume in current window: latest cumulative minus oldest cumulative
        # inside the window.  If window_entries is a subset of the deque the
        # oldest entry in the window is the first element; the matched volume
        # just before the window is the last entry before window_start_ts.
        earliest_in_window = window_entries[0]
        if pre_window_entries:
            # Use the last pre-window entry as the starting reference so we
            # capture volume that accrued from that point to now.
            ref_vol = pre_window_entries[-1][1]
        else:
            ref_vol = earliest_in_window[1]

        latest_vol = hist[-1][1]  # snapshot.matched_volume
        volume_in_window = max(latest_vol - ref_vol, 0.0)

        # ------------------------------------------------------------------
        # Baseline: average volume per window using history before this window.
        # We walk backwards through prior windows to collect up to 5 samples.
        # ------------------------------------------------------------------
        baseline = self._compute_baseline(pre_window_entries, now_ts)

        logger.debug(
            "update %s/%d: vol_window=%.2f baseline=%.2f",
            snapshot.market_id,
            snapshot.selection_id,
            volume_in_window,
            baseline,
        )

        # ------------------------------------------------------------------
        # Spike check
        # ------------------------------------------------------------------
        if baseline <= 0.0:
            # No baseline yet — cannot determine a spike
            return None

        volume_ratio = volume_in_window / baseline

        if volume_in_window < self._min_vol:
            return None

        if volume_ratio < self._spike_mult:
            return None

        # ------------------------------------------------------------------
        # Odds movement check
        # ------------------------------------------------------------------
        odds_before = earliest_in_window[2]  # best_back at start of window
        odds_after = snapshot.best_back

        if odds_before <= 0.0 or odds_after <= 0.0:
            # Cannot compute a meaningful odds move
            return None

        odds_move = odds_before - odds_after  # positive = shortening
        if abs(odds_move) < self._odds_move_thresh:
            logger.debug(
                "update %s/%d: spike detected but odds move %.4f below threshold",
                snapshot.market_id,
                snapshot.selection_id,
                abs(odds_move),
            )
            return None

        # ------------------------------------------------------------------
        # Direction
        # ------------------------------------------------------------------
        direction = "BACK" if odds_move > 0 else "LAY"

        # ------------------------------------------------------------------
        # Confidence
        # ------------------------------------------------------------------
        confidence = self._confidence(volume_ratio, abs(odds_move))

        alert = SmartMoneyAlert(
            market_id=snapshot.market_id,
            selection_id=snapshot.selection_id,
            selection_name=snapshot.selection_name,
            timestamp=now_ts,
            volume_5min=volume_in_window,
            volume_baseline=baseline,
            volume_ratio=volume_ratio,
            odds_before=odds_before,
            odds_after=odds_after,
            odds_move=odds_move,
            direction=direction,
            confidence=confidence,
        )

        self._alerts.append(alert)
        logger.info("SmartMoneyAlert: %s", alert)

        if self._on_alert is not None:
            try:
                self._on_alert(alert)
            except Exception as exc:  # noqa: BLE001
                logger.exception("on_alert callback raised: %s", exc)

        return alert

    def get_alerts(self) -> List[SmartMoneyAlert]:
        """Return all accumulated alerts (does not clear them)."""
        return list(self._alerts)

    def clear_alerts(self) -> None:
        """Clear the accumulated alert list."""
        self._alerts.clear()
        logger.debug("Alert list cleared")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _confidence(self, volume_ratio: float, odds_move: float) -> str:
        """
        HIGH if ratio > 5× and move > 0.15;
        MEDIUM if ratio > 3× and move > 0.05;
        else LOW.
        """
        if volume_ratio > 5.0 and odds_move > 0.15:
            return "HIGH"
        if volume_ratio > 3.0 and odds_move > 0.05:
            return "MEDIUM"
        return "LOW"

    def _compute_baseline(
        self,
        pre_window_entries: List[Tuple[float, float, float]],
        now_ts: float,
    ) -> float:
        """
        Estimate typical volume per window from the historical data that
        predates the current spike window.

        Strategy: divide the pre-window history into sub-windows of the same
        length as self._window, compute the volume transacted in each, then
        return the average (up to 5 sub-windows).

        This gives a like-for-like comparison: how much usually trades in a
        window of this size?
        """
        if not pre_window_entries:
            return 0.0

        samples: List[float] = []
        max_sub_windows = 5

        for i in range(1, max_sub_windows + 1):
            sub_end_ts = now_ts - i * self._window
            sub_start_ts = sub_end_ts - self._window

            # Entries whose timestamp falls within this sub-window
            sub_entries = [
                e for e in pre_window_entries if sub_start_ts <= e[0] < sub_end_ts
            ]

            if len(sub_entries) < 2:
                continue

            vol = sub_entries[-1][1] - sub_entries[0][1]
            if vol > 0:
                samples.append(vol)

        if not samples:
            # Fallback: use the full pre-window range as a single sample
            # and normalise to the window duration.
            total_vol = pre_window_entries[-1][1] - pre_window_entries[0][1]
            elapsed = pre_window_entries[-1][0] - pre_window_entries[0][0]
            if elapsed > 0 and total_vol > 0:
                return total_vol / elapsed * self._window
            return 0.0

        return sum(samples) / len(samples)


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------


def build_tracker(
    on_alert: Optional[Callable[[SmartMoneyAlert], None]] = None,
    window_seconds: float = _DEFAULT_WINDOW_SECONDS,
) -> SmartMoneyTracker:
    """
    Create a SmartMoneyTracker with production defaults.

    Parameters
    ----------
    on_alert       : optional callback invoked whenever an alert fires
    window_seconds : baseline window length (default 300 s)
    """
    return SmartMoneyTracker(
        window_seconds=window_seconds,
        spike_multiplier=_VOLUME_SPIKE_MULTIPLIER,
        min_spike_volume=_MIN_SPIKE_VOLUME,
        odds_move_threshold=_ODDS_MOVE_THRESHOLD,
        on_alert=on_alert,
    )

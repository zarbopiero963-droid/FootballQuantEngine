"""
Sharp vs Soft bookmaker tracker — steam move detection and alert system.

Sharp books (e.g. Pinnacle, Betfair Exchange) reflect professional money
quickly.  Soft books (e.g. Bet365, William Hill) lag by minutes or hours.
When a sharp book's odds drop rapidly, it signals informed money — a "steam
move".  Betting the same side on a soft book before it adjusts gives you
Closing Line Value (CLV) almost by definition.

Architecture
------------
  OddsSnapshot   — immutable record of one odds observation for one fixture
  SteamAlert     — emitted when a sharp move exceeds the threshold
  SharpSoftTracker — stateful tracker; call update() on each new snapshot

Steam detection logic
----------------------
  For each (fixture_id, source) pair the tracker keeps a rolling window of
  the last N snapshots (default: last 30 minutes of history).

  A steam move is triggered when, within the look-back window, the implied
  probability for any outcome has risen by more than `steam_threshold`
  percentage points (default: 4 pp) on a *sharp* source.

  Simultaneously the tracker checks whether soft books have already caught
  up.  If the soft book's implied prob is still below the new sharp level,
  a SoftOpportunity is flagged.

  The "sharp" flag for a source is set via the `sharp_sources` constructor
  argument (default: {"pinnacle", "betfair", "matchbook", "sbo"}).

Usage
-----
    from engine.sharp_soft_tracker import SharpSoftTracker, OddsSnapshot

    tracker = SharpSoftTracker(steam_threshold=0.04, window_seconds=300)

    # Feed snapshots as they arrive (any order within a session)
    tracker.update(OddsSnapshot(fixture_id=123, source="pinnacle",
                                timestamp=1700000000.0,
                                home_odds=2.10, draw_odds=3.40, away_odds=3.20))
    ...

    alerts = tracker.get_alerts(fixture_id=123)
    opps   = tracker.get_opportunities()  # all current soft-book opportunities
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Callable, Deque, Dict, List, Optional, Set

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

_DEFAULT_SHARP = frozenset(
    {"pinnacle", "betfair", "matchbook", "sbo", "ibc", "singbet"}
)
_DEFAULT_WINDOW = 300.0  # 5 minutes
_DEFAULT_STEAM = 0.04  # 4 percentage-point move in implied probability
_MAX_HISTORY = 200  # max snapshots kept per (fixture_id, source)


@dataclass(frozen=True)
class OddsSnapshot:
    """Single odds observation for one fixture from one source."""

    fixture_id: int
    source: str  # bookmaker name (lowercase)
    timestamp: float  # Unix epoch seconds
    home_odds: float
    draw_odds: float
    away_odds: float

    @property
    def home_implied(self) -> float:
        return 1.0 / self.home_odds if self.home_odds > 1.0 else 0.0

    @property
    def draw_implied(self) -> float:
        return 1.0 / self.draw_odds if self.draw_odds > 1.0 else 0.0

    @property
    def away_implied(self) -> float:
        return 1.0 / self.away_odds if self.away_odds > 1.0 else 0.0

    @property
    def overround(self) -> float:
        return self.home_implied + self.draw_implied + self.away_implied

    def margin_removed(self) -> "OddsSnapshot":
        """Return a new snapshot with the bookmaker margin removed (fair odds)."""
        ov = max(self.overround, 1.001)
        return OddsSnapshot(
            fixture_id=self.fixture_id,
            source=self.source,
            timestamp=self.timestamp,
            home_odds=self.home_odds * ov,
            draw_odds=self.draw_odds * ov,
            away_odds=self.away_odds * ov,
        )


@dataclass
class SteamAlert:
    """Emitted when a sharp book moves decisively on a market."""

    fixture_id: int
    source: str  # which sharp book triggered the move
    timestamp: float
    market: str  # "home" | "draw" | "away"
    old_implied: float  # implied prob at window start
    new_implied: float  # current implied prob
    move_pp: float  # move in percentage points (new - old)
    new_odds: float  # current decimal odds on sharp book

    def __str__(self) -> str:
        return (
            f"STEAM [{self.source}] fixture={self.fixture_id} "
            f"market={self.market} move={self.move_pp:+.2%} "
            f"old={self.old_implied:.3f}→new={self.new_implied:.3f} "
            f"odds={self.new_odds:.2f}"
        )


@dataclass
class SoftOpportunity:
    """A soft book that hasn't yet caught up to a sharp steam move."""

    fixture_id: int
    market: str  # "home" | "draw" | "away"
    sharp_implied: float  # new sharp implied prob
    soft_source: str
    soft_implied: float  # current soft implied prob (still lower)
    soft_odds: float  # odds still available on soft book
    edge_pp: float  # sharp_implied - soft_implied
    steam_alert: SteamAlert

    def __str__(self) -> str:
        return (
            f"OPPORTUNITY [{self.soft_source}] fixture={self.fixture_id} "
            f"market={self.market} bet@{self.soft_odds:.2f} "
            f"edge={self.edge_pp:+.2%}"
        )


# ---------------------------------------------------------------------------
# SharpSoftTracker
# ---------------------------------------------------------------------------


class SharpSoftTracker:
    """
    Stateful Sharp vs Soft odds tracker.

    Parameters
    ----------
    sharp_sources    : set of source names treated as sharp books
    steam_threshold  : minimum implied-prob move (in [0,1]) to flag as steam
    window_seconds   : look-back window for steam detection
    on_steam         : optional callback called with SteamAlert on detection
    on_opportunity   : optional callback called with SoftOpportunity
    """

    def __init__(
        self,
        sharp_sources: Optional[Set[str]] = None,
        steam_threshold: float = _DEFAULT_STEAM,
        window_seconds: float = _DEFAULT_WINDOW,
        on_steam: Optional[Callable[[SteamAlert], None]] = None,
        on_opportunity: Optional[Callable[[SoftOpportunity], None]] = None,
    ) -> None:
        self._sharp = sharp_sources or set(_DEFAULT_SHARP)
        self._threshold = steam_threshold
        self._window = window_seconds
        self._on_steam = on_steam
        self._on_opportunity = on_opportunity

        # (fixture_id, source) → deque of OddsSnapshot (chronological)
        self._history: Dict[tuple, Deque[OddsSnapshot]] = defaultdict(
            lambda: deque(maxlen=_MAX_HISTORY)
        )
        # active steam alerts per fixture (cleared when odds recover)
        self._alerts: Dict[int, List[SteamAlert]] = defaultdict(list)
        # latest snapshot per (fixture_id, source)
        self._latest: Dict[tuple, OddsSnapshot] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, snapshot: OddsSnapshot) -> List[SteamAlert]:
        """
        Process a new odds snapshot.

        Returns a list of any SteamAlerts triggered by this update
        (empty list if no steam detected).
        """
        key = (snapshot.fixture_id, snapshot.source.lower())
        self._history[key].append(snapshot)
        self._latest[key] = snapshot

        new_alerts: List[SteamAlert] = []

        if snapshot.source.lower() in self._sharp:
            new_alerts = self._check_steam(snapshot)
            for alert in new_alerts:
                self._alerts[snapshot.fixture_id].append(alert)
                if self._on_steam:
                    self._on_steam(alert)
                # Check soft books for opportunities
                for opp in self._find_opportunities(alert):
                    if self._on_opportunity:
                        self._on_opportunity(opp)

        return new_alerts

    def get_alerts(self, fixture_id: Optional[int] = None) -> List[SteamAlert]:
        """Return steam alerts, optionally filtered to one fixture."""
        if fixture_id is not None:
            return list(self._alerts.get(fixture_id, []))
        return [a for alerts in self._alerts.values() for a in alerts]

    def get_latest(self, fixture_id: int, source: str) -> Optional[OddsSnapshot]:
        """Return the most recent snapshot for a fixture/source pair."""
        return self._latest.get((fixture_id, source.lower()))

    def get_opportunities(self) -> List[SoftOpportunity]:
        """
        Scan all current alerts and return active soft-book opportunities.

        An opportunity is active if the soft book's latest odds still
        give a better implied prob than the sharp book's post-steam level.
        """
        opps = []
        for alerts in self._alerts.values():
            for alert in alerts:
                opps.extend(self._find_opportunities(alert))
        return opps

    def clear_fixture(self, fixture_id: int) -> None:
        """Remove all history and alerts for a finished fixture."""
        keys_to_remove = [k for k in self._history if k[0] == fixture_id]
        for k in keys_to_remove:
            del self._history[k]
            self._latest.pop(k, None)
        self._alerts.pop(fixture_id, None)

    # ------------------------------------------------------------------
    # Steam detection
    # ------------------------------------------------------------------

    def _check_steam(self, snap: OddsSnapshot) -> List[SteamAlert]:
        """
        Compare the current snapshot to the oldest snapshot within the
        look-back window.  Flag steam if implied prob rose by >= threshold.
        """
        key = (snap.fixture_id, snap.source.lower())
        history = self._history[key]
        cutoff = snap.timestamp - self._window

        # Find the oldest snapshot still within the window
        baseline: Optional[OddsSnapshot] = None
        for s in history:
            if s.timestamp >= cutoff:
                baseline = s
                break

        if baseline is None or baseline is snap:
            return []

        alerts = []
        for market, old_imp, new_imp, new_odds in [
            ("home", baseline.home_implied, snap.home_implied, snap.home_odds),
            ("draw", baseline.draw_implied, snap.draw_implied, snap.draw_odds),
            ("away", baseline.away_implied, snap.away_implied, snap.away_odds),
        ]:
            move = new_imp - old_imp
            if move >= self._threshold:
                alert = SteamAlert(
                    fixture_id=snap.fixture_id,
                    source=snap.source,
                    timestamp=snap.timestamp,
                    market=market,
                    old_implied=old_imp,
                    new_implied=new_imp,
                    move_pp=move,
                    new_odds=new_odds,
                )
                alerts.append(alert)

        return alerts

    def _find_opportunities(self, alert: SteamAlert) -> List[SoftOpportunity]:
        """
        For a given steam alert, check all soft books for the same fixture.
        Return opportunities where the soft book hasn't caught up yet.
        """
        opps = []
        fid = alert.fixture_id

        for (fixture_id, source), snap in self._latest.items():
            if fixture_id != fid:
                continue
            if source in self._sharp:
                continue  # skip other sharp books

            soft_imp = getattr(snap, f"{alert.market}_implied", 0.0)
            soft_odds = getattr(snap, f"{alert.market}_odds", 0.0)

            # Opportunity: soft book's implied prob < sharp's new level
            edge = alert.new_implied - soft_imp
            if edge > 0 and soft_odds > 1.0:
                opps.append(
                    SoftOpportunity(
                        fixture_id=fid,
                        market=alert.market,
                        sharp_implied=alert.new_implied,
                        soft_source=source,
                        soft_implied=soft_imp,
                        soft_odds=soft_odds,
                        edge_pp=edge,
                        steam_alert=alert,
                    )
                )
        return opps


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def build_tracker(
    steam_threshold: float = 0.04,
    window_seconds: float = 300.0,
    on_steam: Optional[Callable[[SteamAlert], None]] = None,
    on_opportunity: Optional[Callable[[SoftOpportunity], None]] = None,
) -> SharpSoftTracker:
    """Create a pre-configured SharpSoftTracker with default sharp-book list."""
    return SharpSoftTracker(
        steam_threshold=steam_threshold,
        window_seconds=window_seconds,
        on_steam=on_steam,
        on_opportunity=on_opportunity,
    )

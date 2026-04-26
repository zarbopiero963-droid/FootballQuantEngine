"""
Real-time market pressure analysis for football betting markets.

Detects:
  Steam moves          — rapid odds drop signalling sharp-money action
  Reverse line movement — model side differs from where money is going
  Consensus drift      — divergence between bookmakers over time
  CLV pressure         — closing-line-value trajectory

All analysis is done on a list of odds snapshots sorted by timestamp,
so it works equally with live polling or offline backtesting.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class OddsSnapshot:
    """Single odds observation for a fixture and market."""

    fixture_id: str
    timestamp: float  # Unix epoch seconds
    home_odds: float
    draw_odds: float
    away_odds: float
    source: str = "unknown"  # bookmaker / exchange name


@dataclass
class PressureReport:
    """Pressure analysis result for one fixture."""

    fixture_id: str
    steam_detected: bool = False
    steam_market: str = ""  # "home" | "draw" | "away"
    steam_magnitude: float = 0.0  # fractional odds drop in steam window
    reverse_line_home: bool = False  # model says home but money on away
    reverse_line_away: bool = False
    home_pressure: float = 0.0  # opening→current implied prob shift
    draw_pressure: float = 0.0
    away_pressure: float = 0.0
    sharpness_score: float = 0.0  # composite 0–1 "how sharp is the action"
    clv_trajectory: list = field(default_factory=list)
    snapshots_used: int = 0
    recommendation: str = "NEUTRAL"  # FOLLOW_SHARP | FADE | NEUTRAL


# ---------------------------------------------------------------------------
# Implied probability helpers
# ---------------------------------------------------------------------------


def _implied(odds: float) -> float:
    """Raw implied probability (no vig removal)."""
    return 1.0 / odds if odds > 0 else 0.0


def _vig_free(home_odds: float, draw_odds: float, away_odds: float) -> dict[str, float]:
    """
    Pinnacle/sharp-book vig-free implied probabilities.
    Removes overround by normalising the three implied probabilities.
    """
    ih = _implied(home_odds)
    id_ = _implied(draw_odds)
    ia = _implied(away_odds)
    total = ih + id_ + ia
    if total <= 0:
        return {"home": 1 / 3, "draw": 1 / 3, "away": 1 / 3}
    return {
        "home": ih / total,
        "draw": id_ / total,
        "away": ia / total,
    }


def _overround(home_odds: float, draw_odds: float, away_odds: float) -> float:
    """Bookmaker overround (margin) as a fraction."""
    return _implied(home_odds) + _implied(draw_odds) + _implied(away_odds) - 1.0


# ---------------------------------------------------------------------------
# Steam move detection
# ---------------------------------------------------------------------------

_STEAM_THRESHOLD_PCT: float = 0.04  # 4% implied-prob move counts as steam
_STEAM_WINDOW_SECS: float = 300.0  # within 5 minutes


def _detect_steam(
    snapshots: list[OddsSnapshot],
) -> tuple[bool, str, float]:
    """
    Scan sorted snapshots for a rapid odds compression on any outcome.

    Returns (detected, market, magnitude).
    """
    if len(snapshots) < 2:
        return False, "", 0.0

    best_market = ""
    best_mag = 0.0
    found = False

    for i in range(1, len(snapshots)):
        prev = snapshots[i - 1]
        curr = snapshots[i]

        dt = curr.timestamp - prev.timestamp
        if dt <= 0 or dt > _STEAM_WINDOW_SECS:
            continue

        p_prev = _vig_free(prev.home_odds, prev.draw_odds, prev.away_odds)
        p_curr = _vig_free(curr.home_odds, curr.draw_odds, curr.away_odds)

        for mkt in ("home", "draw", "away"):
            shift = p_curr[mkt] - p_prev[mkt]
            if shift >= _STEAM_THRESHOLD_PCT and shift > best_mag:
                best_mag = shift
                best_market = mkt
                found = True

    return found, best_market, round(best_mag, 4)


# ---------------------------------------------------------------------------
# Reverse line movement
# ---------------------------------------------------------------------------


def _detect_reverse_line(
    snapshots: list[OddsSnapshot],
    model_home_prob: float | None = None,
) -> tuple[bool, bool]:
    """
    Reverse line movement (RLM): odds move against expected public side.

    Logic: the public typically bets home. If home odds shorten (sharp
    money on home) but our model also likes home → no RLM.
    If home odds lengthen (public on home, but sharp money on away) →
    potential RLM on away.

    Returns (rlm_home, rlm_away).
    """
    if len(snapshots) < 2:
        return False, False

    first = snapshots[0]
    last = snapshots[-1]

    p_open = _vig_free(first.home_odds, first.draw_odds, first.away_odds)
    p_close = _vig_free(last.home_odds, last.draw_odds, last.away_odds)

    home_drift = p_close["home"] - p_open["home"]
    away_drift = p_close["away"] - p_open["away"]

    # If home odds shortened (money on home) but model doubts home → RLM away
    rlm_home = (away_drift > 0.02) and (
        model_home_prob is not None and model_home_prob < p_open["home"]
    )
    rlm_away = (home_drift > 0.02) and (
        model_home_prob is not None and model_home_prob > p_open["home"]
    )

    return rlm_home, rlm_away


# ---------------------------------------------------------------------------
# Sharpness score
# ---------------------------------------------------------------------------


def _sharpness_score(
    snapshots: list[OddsSnapshot],
    steam_mag: float,
    overround_drop: float,
) -> float:
    """
    Composite 0–1 sharpness indicator.

    Factors:
      - Steam magnitude (rapid odds compression)
      - Overround reduction (sharp books cut margin under pressure)
      - Velocity of odds movement per unit time
    """
    if len(snapshots) < 2:
        return 0.0

    first = snapshots[0]
    last = snapshots[-1]
    dt = max(last.timestamp - first.timestamp, 1.0)

    p_open = _vig_free(first.home_odds, first.draw_odds, first.away_odds)
    p_close = _vig_free(last.home_odds, last.draw_odds, last.away_odds)

    total_shift = sum(abs(p_close[m] - p_open[m]) for m in ("home", "draw", "away"))
    velocity = total_shift / (dt / 3600.0 + 1e-6)  # per hour

    # Normalise components
    steam_score = min(steam_mag / 0.10, 1.0)
    vig_score = min(max(overround_drop / 0.02, 0.0), 1.0)
    velocity_score = min(velocity / 0.20, 1.0)

    return round(0.4 * steam_score + 0.35 * vig_score + 0.25 * velocity_score, 4)


# ---------------------------------------------------------------------------
# CLV trajectory
# ---------------------------------------------------------------------------


def _clv_trajectory(
    snapshots: list[OddsSnapshot],
    model_prob: float,
    market: str = "home",
) -> list[dict]:
    """
    Track CLV (Closing Line Value) at each snapshot relative to model probability.

    CLV_t = model_prob - implied_prob_t

    A positive trajectory means the model is ahead of the market:
    the model liked an outcome before the market agreed.
    """
    traj = []
    for snap in snapshots:
        probs = _vig_free(snap.home_odds, snap.draw_odds, snap.away_odds)
        clv = model_prob - probs.get(market, 0.5)
        traj.append(
            {
                "timestamp": snap.timestamp,
                "clv": round(clv, 5),
                "impl_prob": round(probs.get(market, 0.5), 5),
            }
        )
    return traj


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def calculate_market_pressure(
    open_odds: float,
    current_odds: float,
) -> float:
    """
    Simple fractional market pressure scalar (backwards-compatible function).

    Positive → odds shortened (money on this outcome).
    """
    if open_odds <= 0:
        return 0.0
    open_ip = _implied(open_odds)
    current_ip = _implied(current_odds)
    return round(current_ip - open_ip, 6)


class MarketPressureLive:
    """
    Full market pressure analyser operating on a list of OddsSnapshot objects.

    Usage
    -----
    analyser = MarketPressureLive()
    report   = analyser.analyse(fixture_id, snapshots, model_home_prob=0.52)
    """

    def __init__(
        self,
        steam_threshold: float = _STEAM_THRESHOLD_PCT,
        steam_window: float = _STEAM_WINDOW_SECS,
    ) -> None:
        self._steam_thr = steam_threshold
        self._steam_win = steam_window

    def analyse(
        self,
        fixture_id: str,
        snapshots: list[OddsSnapshot],
        model_home_prob: float | None = None,
    ) -> PressureReport:
        """
        Run all pressure detectors on the snapshot series.

        Parameters
        ----------
        fixture_id      : unique fixture identifier
        snapshots       : list of OddsSnapshot, sorted ascending by timestamp
        model_home_prob : optional model probability for RLM / CLV analysis

        Returns
        -------
        PressureReport dataclass with all computed signals.
        """
        report = PressureReport(fixture_id=fixture_id)

        if not snapshots:
            return report

        snaps = sorted(snapshots, key=lambda s: s.timestamp)
        report.snapshots_used = len(snaps)

        # Opening and closing snapshots
        first, last = snaps[0], snaps[-1]

        # Implied probability shift (opening → current)
        p_open = _vig_free(first.home_odds, first.draw_odds, first.away_odds)
        p_close = _vig_free(last.home_odds, last.draw_odds, last.away_odds)

        report.home_pressure = round(p_close["home"] - p_open["home"], 5)
        report.draw_pressure = round(p_close["draw"] - p_open["draw"], 5)
        report.away_pressure = round(p_close["away"] - p_open["away"], 5)

        # Steam detection
        steam, steam_mkt, steam_mag = _detect_steam(snaps)
        report.steam_detected = steam
        report.steam_market = steam_mkt
        report.steam_magnitude = steam_mag

        # Reverse line movement
        rlm_h, rlm_a = _detect_reverse_line(snaps, model_home_prob)
        report.reverse_line_home = rlm_h
        report.reverse_line_away = rlm_a

        # Overround change
        or_open = _overround(first.home_odds, first.draw_odds, first.away_odds)
        or_close = _overround(last.home_odds, last.draw_odds, last.away_odds)
        overround_drop = or_open - or_close

        # Sharpness score
        report.sharpness_score = _sharpness_score(snaps, steam_mag, overround_drop)

        # CLV trajectory (for model's favoured market)
        if model_home_prob is not None:
            if model_home_prob > 0.50:
                main_market = "home"
            elif model_home_prob < 0.38:
                main_market = "away"
            else:
                main_market = "draw"
            report.clv_trajectory = _clv_trajectory(snaps, model_home_prob, main_market)

        # Recommendation
        if report.steam_detected and report.sharpness_score > 0.6:
            report.recommendation = "FOLLOW_SHARP"
        elif (rlm_h or rlm_a) and report.sharpness_score > 0.4:
            report.recommendation = "FADE"
        else:
            report.recommendation = "NEUTRAL"

        return report

    def analyse_from_dicts(
        self,
        fixture_id: str,
        snapshots_raw: list[dict],
        model_home_prob: float | None = None,
    ) -> PressureReport:
        """
        Convenience wrapper accepting dicts with keys:
          timestamp, home_odds, draw_odds, away_odds, [source].
        """
        snaps = [
            OddsSnapshot(
                fixture_id=fixture_id,
                timestamp=float(d["timestamp"]),
                home_odds=float(d["home_odds"]),
                draw_odds=float(d["draw_odds"]),
                away_odds=float(d["away_odds"]),
                source=str(d.get("source", "unknown")),
            )
            for d in snapshots_raw
        ]
        return self.analyse(fixture_id, snaps, model_home_prob)

    @staticmethod
    def vig_free_probs(
        home_odds: float, draw_odds: float, away_odds: float
    ) -> dict[str, float]:
        """Public: vig-free implied probabilities from three-way odds."""
        return _vig_free(home_odds, draw_odds, away_odds)

    @staticmethod
    def overround(home_odds: float, draw_odds: float, away_odds: float) -> float:
        """Bookmaker overround (vig) as a fraction."""
        return round(_overround(home_odds, draw_odds, away_odds), 6)

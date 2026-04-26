"""
In-Play / Live Betting Edge Engine.

Computes real-time betting value using the current match state — elapsed
minutes, current score, and red cards — combined with pre-match Poisson
parameters.

How it works
------------
1. Scale pre-match lambdas by the fraction of the game remaining.
2. Apply red-card penalties (approx −0.3 goals/90min per card).
3. Build a Poisson score matrix for *remaining* goals in the match.
4. Convolve with the current score to obtain conditional 1X2 / O/U probs.
5. Compare model probabilities against live market odds to find value bets.

Red-card model
--------------
Each red card reduces the affected team's scoring rate by 0.3 goals/90min
(scaled by remaining fraction).  Minimum λ per team is capped at 0.05.

Minute-scaling
--------------
  remaining_frac = max(90 − elapsed, 0) / 90

  λ_home_rem = λ_home_prematch × remaining_frac × penalty_home
  λ_away_rem = λ_away_prematch × remaining_frac × penalty_away

  where penalty = max(0.1, 1 − 0.3 × red_cards)

Usage
-----
    from engine.live_edge_engine import LiveEdgeEngine, MatchState

    engine = LiveEdgeEngine(lambda_home=1.6, lambda_away=1.0)

    state = MatchState(
        elapsed=60,
        home_score=1,
        away_score=0,
        red_cards_home=0,
        red_cards_away=1,
    )

    result = engine.compute(state)
    print(result.p_home_win, result.fair_odds_home)

    edge = engine.edge_vs_market(state, market_home=2.10, market_draw=3.20, market_away=3.80)
    # {"market": "home", "edge": 0.07, "fair_odds": 1.92, "market_odds": 2.10}
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

_MAX_GOALS = 10  # truncation for remaining-goals matrix
_RC_PENALTY = 0.30  # goals/90min reduction per red card
_MIN_LAMBDA = 0.05  # floor for remaining λ
_MIN_EDGE = 0.02  # default minimum edge to flag a value bet


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class MatchState:
    """Snapshot of the current in-play match state."""

    elapsed: float  # minutes played (0–90+)
    home_score: int  # goals scored by home so far
    away_score: int  # goals scored by away so far
    red_cards_home: int = 0
    red_cards_away: int = 0

    @property
    def remaining_minutes(self) -> float:
        return max(90.0 - self.elapsed, 0.0)

    @property
    def remaining_frac(self) -> float:
        return self.remaining_minutes / 90.0


@dataclass
class LiveEdgeResult:
    """
    Model probabilities and fair odds for the current live state.

    All probabilities are conditional on the current score and remaining time.
    """

    elapsed: float
    home_score: int
    away_score: int
    remaining_frac: float

    # Remaining-goals lambdas used
    lambda_home_rem: float
    lambda_away_rem: float

    # Conditional 1X2 probabilities (for the full-time result)
    p_home_win: float
    p_draw: float
    p_away_win: float

    # Fair decimal odds (1 / p, floored at 1.01)
    fair_odds_home: float
    fair_odds_draw: float
    fair_odds_away: float

    # Over/Under 0.5 additional goals (useful live)
    p_over_0_5: float  # probability at least 1 more goal is scored
    p_under_0_5: float

    def __str__(self) -> str:
        return (
            f"LiveEdge min={self.elapsed:.0f}' score={self.home_score}-{self.away_score} "
            f"rem_frac={self.remaining_frac:.2f} "
            f"1X2=({self.p_home_win:.3f}/{self.p_draw:.3f}/{self.p_away_win:.3f}) "
            f"odds=({self.fair_odds_home:.2f}/{self.fair_odds_draw:.2f}/{self.fair_odds_away:.2f})"
        )


@dataclass
class LiveValueBet:
    """A value-bet opportunity identified against live market odds."""

    market: str  # "home" | "draw" | "away"
    edge: float  # model_prob − 1/market_odds
    model_prob: float
    fair_odds: float
    market_odds: float
    elapsed: float
    home_score: int
    away_score: int

    def __str__(self) -> str:
        return (
            f"LIVE VALUE [{self.market}] min={self.elapsed:.0f}' "
            f"score={self.home_score}-{self.away_score} "
            f"edge={self.edge:+.2%} fair={self.fair_odds:.2f} mkt={self.market_odds:.2f}"
        )


# ---------------------------------------------------------------------------
# LiveEdgeEngine
# ---------------------------------------------------------------------------


class LiveEdgeEngine:
    """
    Real-time Poisson-based edge calculator for in-play markets.

    Parameters
    ----------
    lambda_home     : pre-match expected home goals (full 90 minutes)
    lambda_away     : pre-match expected away goals
    max_goals       : score-matrix truncation for remaining goals (default 10)
    rc_penalty      : goals/90min lost per red card (default 0.30)
    """

    def __init__(
        self,
        lambda_home: float,
        lambda_away: float,
        max_goals: int = _MAX_GOALS,
        rc_penalty: float = _RC_PENALTY,
    ) -> None:
        self.lambda_home = max(0.05, lambda_home)
        self.lambda_away = max(0.05, lambda_away)
        self._max = max_goals
        self._rc_penalty = rc_penalty

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(self, state: MatchState) -> LiveEdgeResult:
        """
        Compute conditional live probabilities for the given match state.

        Returns a LiveEdgeResult with 1X2 probabilities and fair odds.
        """
        lh_rem, la_rem = self._remaining_lambdas(state)

        # Score matrix for remaining goals
        rem_matrix = _build_score_matrix(lh_rem, la_rem, self._max)

        # Aggregate into conditional 1X2
        p_home, p_draw, p_away = _conditional_1x2(
            rem_matrix, state.home_score, state.away_score, self._max
        )

        # Over/Under 0.5 remaining goals
        p_over_05 = _p_at_least_one_goal(lh_rem, la_rem)
        p_under_05 = 1.0 - p_over_05

        return LiveEdgeResult(
            elapsed=state.elapsed,
            home_score=state.home_score,
            away_score=state.away_score,
            remaining_frac=state.remaining_frac,
            lambda_home_rem=lh_rem,
            lambda_away_rem=la_rem,
            p_home_win=p_home,
            p_draw=p_draw,
            p_away_win=p_away,
            fair_odds_home=_inv(p_home),
            fair_odds_draw=_inv(p_draw),
            fair_odds_away=_inv(p_away),
            p_over_0_5=p_over_05,
            p_under_0_5=p_under_05,
        )

    def edge_vs_market(
        self,
        state: MatchState,
        market_home: float,
        market_draw: float,
        market_away: float,
        min_edge: float = _MIN_EDGE,
    ) -> List[LiveValueBet]:
        """
        Compare model probabilities against live market odds.

        Returns a list of LiveValueBet objects where the model edge exceeds
        min_edge.  An empty list means no value found.
        """
        result = self.compute(state)
        bets: List[LiveValueBet] = []

        for market, model_p, fair_o, mkt_o in [
            ("home", result.p_home_win, result.fair_odds_home, market_home),
            ("draw", result.p_draw, result.fair_odds_draw, market_draw),
            ("away", result.p_away_win, result.fair_odds_away, market_away),
        ]:
            if mkt_o <= 1.0:
                continue
            implied = 1.0 / mkt_o
            edge = model_p - implied
            if edge >= min_edge:
                bets.append(
                    LiveValueBet(
                        market=market,
                        edge=edge,
                        model_prob=model_p,
                        fair_odds=fair_o,
                        market_odds=mkt_o,
                        elapsed=state.elapsed,
                        home_score=state.home_score,
                        away_score=state.away_score,
                    )
                )

        bets.sort(key=lambda b: b.edge, reverse=True)
        return bets

    def kelly_stake(
        self,
        model_prob: float,
        market_odds: float,
        fraction: float = 0.25,
    ) -> float:
        """
        Fractional Kelly stake as a fraction of bankroll.

        fraction=0.25 → quarter-Kelly (conservative default).
        Returns 0.0 if Kelly is negative (no edge).
        """
        b = market_odds - 1.0
        if b <= 0:
            return 0.0
        q = 1.0 - model_prob
        f_star = (model_prob * b - q) / b
        return max(0.0, f_star * fraction)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _remaining_lambdas(self, state: MatchState) -> Tuple[float, float]:
        """Scale pre-match lambdas for remaining time and red cards."""
        frac = state.remaining_frac

        pen_home = max(0.1, 1.0 - self._rc_penalty * state.red_cards_home)
        pen_away = max(0.1, 1.0 - self._rc_penalty * state.red_cards_away)

        lh = max(_MIN_LAMBDA, self.lambda_home * frac * pen_home)
        la = max(_MIN_LAMBDA, self.lambda_away * frac * pen_away)
        return lh, la


# ---------------------------------------------------------------------------
# Score-matrix helpers
# ---------------------------------------------------------------------------


def _build_score_matrix(
    lh: float, la: float, max_goals: int
) -> Dict[Tuple[int, int], float]:
    """Poisson joint PMF for (home_goals, away_goals), renormalized."""
    matrix: Dict[Tuple[int, int], float] = {}
    total = 0.0
    for hg in range(max_goals + 1):
        for ag in range(max_goals + 1):
            p = _poisson_pmf(hg, lh) * _poisson_pmf(ag, la)
            matrix[(hg, ag)] = p
            total += p
    if total > 0:
        matrix = {k: v / total for k, v in matrix.items()}
    return matrix


def _conditional_1x2(
    rem_matrix: Dict[Tuple[int, int], float],
    home_now: int,
    away_now: int,
    max_goals: int,
) -> Tuple[float, float, float]:
    """
    Compute P(home win | current score) by marginalising over remaining goals.

    For each (add_h, add_a) remaining goals:
      final_home = home_now + add_h
      final_away = away_now + add_a
    Then classify as 1X2.
    """
    p_home = p_draw = p_away = 0.0

    for (add_h, add_a), prob in rem_matrix.items():
        fh = home_now + add_h
        fa = away_now + add_a
        if fh > fa:
            p_home += prob
        elif fh == fa:
            p_draw += prob
        else:
            p_away += prob

    total = p_home + p_draw + p_away
    if total <= 0:
        return 1 / 3, 1 / 3, 1 / 3

    return p_home / total, p_draw / total, p_away / total


def _p_at_least_one_goal(lh: float, la: float) -> float:
    """P(at least 1 more goal) = 1 - P(0 home goals) × P(0 away goals)."""
    p_zero_h = math.exp(-lh)
    p_zero_a = math.exp(-la)
    return 1.0 - p_zero_h * p_zero_a


def _poisson_pmf(k: int, lam: float) -> float:
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    try:
        return math.exp(-lam) * (lam**k) / math.factorial(k)
    except (OverflowError, ValueError):
        return 0.0


def _inv(p: float, floor: float = 1.01) -> float:
    if p <= 0:
        return 999.0
    return max(floor, 1.0 / p)


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def compute_live_edge(
    lambda_home: float,
    lambda_away: float,
    elapsed: float,
    home_score: int,
    away_score: int,
    red_cards_home: int = 0,
    red_cards_away: int = 0,
    market_home: Optional[float] = None,
    market_draw: Optional[float] = None,
    market_away: Optional[float] = None,
    min_edge: float = _MIN_EDGE,
) -> Tuple[LiveEdgeResult, List[LiveValueBet]]:
    """
    One-shot convenience: compute live edge result and optional value bets.

    If market odds are supplied, value bets are returned; otherwise the
    second element of the tuple is an empty list.
    """
    engine = LiveEdgeEngine(lambda_home=lambda_home, lambda_away=lambda_away)
    state = MatchState(
        elapsed=elapsed,
        home_score=home_score,
        away_score=away_score,
        red_cards_home=red_cards_home,
        red_cards_away=red_cards_away,
    )
    result = engine.compute(state)

    bets: List[LiveValueBet] = []
    if market_home is not None and market_draw is not None and market_away is not None:
        bets = engine.edge_vs_market(
            state,
            market_home=market_home,
            market_draw=market_draw,
            market_away=market_away,
            min_edge=min_edge,
        )

    return result, bets

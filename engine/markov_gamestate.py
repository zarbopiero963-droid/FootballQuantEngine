"""
Markov Game-State Transition Engine
=====================================

Models football as a discrete Markov chain over (score, time-period) states.
Unlike Poisson models that assume a flat 90-minute rate, this engine knows that
Atletico Madrid leading 1-0 at minute 60 has a completely different attack
intensity than when level at 0-0.

States
------
  (home_goals, away_goals, period)  where period ∈ {0..5} (each = 15 min)

Intensity multipliers
---------------------
  Score diff (home perspective): ≥+2 → 0.55 (protecting lead), 0 → 1.00, ≤-2 → 1.55
  Period:  0-15 → 0.85, 15-30 → 1.00, 30-45 → 1.05, 45-60 → 0.90, 60-75 → 1.05, 75-90 → 1.25

Usage
-----
    from engine.markov_gamestate import analyse_live_state

    session = analyse_live_state(
        home_goals=1, away_goals=0, minute=62,
        lambda_home=1.45, lambda_away=1.20,
        market_odds={"next_home": 2.10, "next_away": 3.80},
    )
    for sig in session.signals:
        print(sig)
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_MAX_GOALS = 4  # score capped at 4 per team for matrix size
_N_PERIODS = 6  # 6 × 15-min periods
_PERIOD_MINUTES = 15  # minutes per period


# ---------------------------------------------------------------------------
# Intensity tables
# ---------------------------------------------------------------------------

_HOME_INTENSITY: Dict[int, float] = {
    2: 0.55,
    1: 0.75,
    0: 1.00,
    -1: 1.35,
    -2: 1.55,
}

_PERIOD_MULTIPLIERS: List[float] = [0.85, 1.00, 1.05, 0.90, 1.05, 1.25]


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class GameState:
    home_goals: int
    away_goals: int
    period: int  # 0-5

    @property
    def score_diff(self) -> int:
        return self.home_goals - self.away_goals

    @property
    def minute_start(self) -> int:
        return self.period * _PERIOD_MINUTES

    @property
    def minute_end(self) -> int:
        return (self.period + 1) * _PERIOD_MINUTES

    @property
    def label(self) -> str:
        return f"{self.home_goals}-{self.away_goals} @ {self.minute_start}-{self.minute_end}'"

    def is_terminal(self) -> bool:
        return self.period >= _N_PERIODS


@dataclass
class TransitionProbabilities:
    from_state: GameState
    lambda_home: float
    lambda_away: float
    p_home_scores: float
    p_away_scores: float
    p_no_goal: float
    p_home_win_from_here: float
    p_draw_from_here: float
    p_away_win_from_here: float


@dataclass
class LiveBettingSignal:
    state: GameState
    signal_type: str
    model_prob: float
    rationale: str
    urgency: str  # "HIGH" | "MEDIUM" | "LOW"

    def __str__(self) -> str:
        return (
            f"[{self.urgency}] {self.signal_type} | "
            f"p={self.model_prob:.3f} | {self.state.label} | {self.rationale}"
        )


@dataclass
class MarkovSession:
    initial_state: GameState
    lambda_home_base: float
    lambda_away_base: float
    next_goal_home_prob: float
    next_goal_away_prob: float
    p_home_win: float
    p_draw: float
    p_away_win: float
    signals: List[LiveBettingSignal] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class MarkovGameEngine:
    """
    Transition-based live football model.

    Parameters
    ----------
    lambda_home_base, lambda_away_base : float
        Pre-match expected goals per 90 min.
    team_style_home, team_style_away : float
        Multiplier on the tactical intensity curve: >1 = more attacking than
        the average team in that score-state (e.g. 1.15 for Bayern Munich);
        <1 = more defensive (e.g. 0.80 for Atletico Madrid).
    """

    def __init__(
        self,
        lambda_home_base: float,
        lambda_away_base: float,
        max_goals: int = _MAX_GOALS,
        team_style_home: float = 1.0,
        team_style_away: float = 1.0,
    ) -> None:
        self._lh = lambda_home_base
        self._la = lambda_away_base
        self._max_goals = max_goals
        self._style_h = team_style_home
        self._style_a = team_style_away

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transition_probs(self, state: GameState) -> TransitionProbabilities:
        """Compute one-period transition probabilities from *state*."""
        eff_lh = self._effective_lambda(self._lh, state, is_home=True)
        eff_la = self._effective_lambda(self._la, state, is_home=False)

        # P(k≥1 in one period) = 1 - exp(-λ * period_fraction)
        frac = _PERIOD_MINUTES / 90.0
        p_h = 1.0 - math.exp(-eff_lh * frac)
        p_a = 1.0 - math.exp(-eff_la * frac)
        # Approximate independence (rare double-goal in 15 min)
        p_none = (1.0 - p_h) * (1.0 - p_a)
        p_both = p_h * p_a
        # Normalise: if both score, we attribute to the "first" randomly
        p_home_net = p_h * (1.0 - p_a) + p_both * 0.5
        p_away_net = p_a * (1.0 - p_h) + p_both * 0.5

        # Forward simulation for final outcome
        wdl = self._simulate_from(state, n=20_000)

        return TransitionProbabilities(
            from_state=state,
            lambda_home=eff_lh,
            lambda_away=eff_la,
            p_home_scores=p_home_net,
            p_away_scores=p_away_net,
            p_no_goal=p_none,
            p_home_win_from_here=wdl[0],
            p_draw_from_here=wdl[1],
            p_away_win_from_here=wdl[2],
        )

    def simulate_from(
        self, state: GameState, n_simulations: int = 50_000
    ) -> MarkovSession:
        """Monte Carlo simulation from *state* through remaining periods."""
        wdl = self._simulate_from(state, n_simulations)
        tp = self.transition_probs(state)

        session = MarkovSession(
            initial_state=state,
            lambda_home_base=self._lh,
            lambda_away_base=self._la,
            next_goal_home_prob=tp.p_home_scores,
            next_goal_away_prob=tp.p_away_scores,
            p_home_win=wdl[0],
            p_draw=wdl[1],
            p_away_win=wdl[2],
        )
        session.signals = self.generate_signals(state)
        return session

    def next_goal_probs(self, state: GameState) -> Tuple[float, float, float]:
        """Returns (p_home_next, p_away_next, p_no_goal_soon)."""
        tp = self.transition_probs(state)
        return tp.p_home_scores, tp.p_away_scores, tp.p_no_goal

    def generate_signals(
        self,
        state: GameState,
        market_odds: Optional[Dict[str, float]] = None,
    ) -> List[LiveBettingSignal]:
        """Generate live betting signals based on current state."""
        signals: List[LiveBettingSignal] = []
        tp = self.transition_probs(state)
        diff = state.score_diff

        # ---- Next goal market ----
        if market_odds and "next_home" in market_odds:
            book_implied = 1.0 / market_odds["next_home"]
            if tp.p_home_scores > book_implied * 1.08:
                signals.append(
                    LiveBettingSignal(
                        state=state,
                        signal_type="NEXT_GOAL_HOME",
                        model_prob=tp.p_home_scores,
                        rationale=(
                            f"Model {tp.p_home_scores:.3f} vs book implied {book_implied:.3f} "
                            f"(+{(tp.p_home_scores / book_implied - 1) * 100:.1f}%)"
                        ),
                        urgency="HIGH"
                        if tp.p_home_scores > book_implied * 1.15
                        else "MEDIUM",
                    )
                )

        if market_odds and "next_away" in market_odds:
            book_implied = 1.0 / market_odds["next_away"]
            if tp.p_away_scores > book_implied * 1.08:
                signals.append(
                    LiveBettingSignal(
                        state=state,
                        signal_type="NEXT_GOAL_AWAY",
                        model_prob=tp.p_away_scores,
                        rationale=(
                            f"Model {tp.p_away_scores:.3f} vs book implied {book_implied:.3f} "
                            f"(+{(tp.p_away_scores / book_implied - 1) * 100:.1f}%)"
                        ),
                        urgency="HIGH"
                        if tp.p_away_scores > book_implied * 1.15
                        else "MEDIUM",
                    )
                )

        # ---- 1X2 outcome signals ----
        for outcome, prob, key in [
            ("BACK_HOME", tp.p_home_win_from_here, "home_win"),
            ("BACK_DRAW", tp.p_draw_from_here, "draw"),
            ("BACK_AWAY", tp.p_away_win_from_here, "away_win"),
        ]:
            if market_odds and key in market_odds:
                book_imp = 1.0 / market_odds[key]
                if prob > book_imp * 1.06:
                    signals.append(
                        LiveBettingSignal(
                            state=state,
                            signal_type=outcome,
                            model_prob=prob,
                            rationale=f"p={prob:.3f} > book {book_imp:.3f}",
                            urgency="MEDIUM",
                        )
                    )

        # ---- Tactical signals (no odds needed) ----
        if diff >= 2 and state.period >= 4:
            signals.append(
                LiveBettingSignal(
                    state=state,
                    signal_type="NEXT_GOAL_AWAY",
                    model_prob=tp.p_away_scores,
                    rationale=f"Winning team likely defensive (diff={diff}) — away goal spike",
                    urgency="LOW",
                )
            )
        if diff == 0 and state.period >= 5:
            signals.append(
                LiveBettingSignal(
                    state=state,
                    signal_type="NEXT_GOAL_HOME",
                    model_prob=tp.p_home_scores,
                    rationale="Level in 75-90': home teams historically push harder (crowd factor)",
                    urgency="LOW",
                )
            )

        return signals

    def team_comeback_prob(self, state: GameState) -> Dict[str, float]:
        """P(team equalises | currently trailing) and P(team wins | trailing)."""
        wdl = self._simulate_from(state, n=30_000)
        diff = state.score_diff
        result: Dict[str, float] = {}
        if diff < 0:
            result["home_equalises"] = wdl[0] + wdl[1]
            result["home_wins"] = wdl[0]
        elif diff > 0:
            result["away_equalises"] = wdl[2] + wdl[1]
            result["away_wins"] = wdl[2]
        else:
            result["home_wins"] = wdl[0]
            result["draw"] = wdl[1]
            result["away_wins"] = wdl[2]
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _effective_lambda(
        self, base_lambda: float, state: GameState, is_home: bool
    ) -> float:
        """Apply state-intensity and period multipliers."""
        diff = self._clamp(state.score_diff if is_home else -state.score_diff)
        intensity = _HOME_INTENSITY.get(diff, 1.00)
        period_mult = _PERIOD_MULTIPLIERS[min(state.period, _N_PERIODS - 1)]
        style = self._style_h if is_home else self._style_a
        return max(0.05, base_lambda * intensity * period_mult * style)

    def _clamp(self, diff: int) -> int:
        return max(-2, min(2, diff))

    def _poisson_pmf(self, k: int, lam: float) -> float:
        if lam <= 0.0:
            return 1.0 if k == 0 else 0.0
        return math.exp(-lam + k * math.log(lam) - math.lgamma(k + 1))

    def _sample_goals(self, lam: float) -> int:
        """Sample Poisson(lam) using uniform transform."""
        threshold = math.exp(-lam)
        prod = random.random()
        k = 0
        while prod > threshold and k < 10:
            prod *= random.random()
            k += 1
        return k

    def _simulate_from(self, state: GameState, n: int) -> Tuple[float, float, float]:
        """Monte Carlo: simulate remaining periods, return (p_home_win, p_draw, p_away_win)."""
        home_wins = 0
        draws = 0
        away_wins = 0
        frac = _PERIOD_MINUTES / 90.0

        for _ in range(n):
            hg = state.home_goals
            ag = state.away_goals
            for period in range(state.period, _N_PERIODS):
                sim_state = GameState(hg, ag, period)
                eff_lh = self._effective_lambda(self._lh, sim_state, True)
                eff_la = self._effective_lambda(self._la, sim_state, False)
                # Sample goals this period
                hg += self._sample_goals(eff_lh * frac)
                ag += self._sample_goals(eff_la * frac)

            if hg > ag:
                home_wins += 1
            elif hg == ag:
                draws += 1
            else:
                away_wins += 1

        return home_wins / n, draws / n, away_wins / n


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------


def analyse_live_state(
    home_goals: int,
    away_goals: int,
    minute: int,
    lambda_home: float,
    lambda_away: float,
    market_odds: Optional[Dict[str, float]] = None,
    team_style_home: float = 1.0,
    team_style_away: float = 1.0,
) -> MarkovSession:
    """
    One-shot: build engine and analyse the current live state.

    Parameters
    ----------
    home_goals, away_goals : current score
    minute : current match minute (0-90)
    lambda_home, lambda_away : pre-match expected goals per 90 min
    market_odds : optional live market prices to compare against
        keys: "next_home", "next_away", "home_win", "draw", "away_win"
    """
    period = min(int(minute / _PERIOD_MINUTES), _N_PERIODS - 1)
    state = GameState(
        home_goals=min(home_goals, _MAX_GOALS),
        away_goals=min(away_goals, _MAX_GOALS),
        period=period,
    )
    engine = MarkovGameEngine(
        lambda_home_base=lambda_home,
        lambda_away_base=lambda_away,
        team_style_home=team_style_home,
        team_style_away=team_style_away,
    )
    session = engine.simulate_from(state, n_simulations=30_000)
    if market_odds:
        session.signals = engine.generate_signals(state, market_odds)
    return session

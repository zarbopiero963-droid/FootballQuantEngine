"""
Pure-math Poisson helpers for the Bayesian Live Engine.

Extracted from BayesianLiveEngine to keep bayesian_live.py under 400 LOC.
All functions are stateless and operate only on numeric parameters.
"""

from __future__ import annotations

import math
from typing import Tuple

from engine.bayesian_types import _MAX_GOALS


def poisson_pmf(k: int, lam: float) -> float:
    """Compute P(X = k) for X ~ Poisson(lam)."""
    if lam <= 0.0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def compute_win_probs(
    rem_lh: float,
    rem_la: float,
    home_score: int,
    away_score: int,
) -> Tuple[float, float, float]:
    """
    Return (p_home_win, p_draw, p_away_win) given remaining Poisson rates
    and the current score.

    Iterates over all (home_goals, away_goals) pairs up to _MAX_GOALS to
    compute the full joint probability mass for each 1x2 outcome.
    """
    p_hw = p_d = p_aw = 0.0
    for gh in range(_MAX_GOALS + 1):
        p_gh = poisson_pmf(gh, rem_lh)
        for ga in range(_MAX_GOALS + 1):
            p_ga = poisson_pmf(ga, rem_la)
            joint = p_gh * p_ga
            total_home = home_score + gh
            total_away = away_score + ga
            if total_home > total_away:
                p_hw += joint
            elif total_home == total_away:
                p_d += joint
            else:
                p_aw += joint
    return p_hw, p_d, p_aw


def compute_over_under(
    rem_lh: float,
    rem_la: float,
    already_scored: int,
    threshold: float,
) -> Tuple[float, float]:
    """
    Return (p_over, p_under) for total match goals relative to *threshold*.

    *already_scored* is the number of goals already in the match.
    """
    p_over = 0.0
    for gh in range(_MAX_GOALS + 1):
        p_gh = poisson_pmf(gh, rem_lh)
        for ga in range(_MAX_GOALS + 1):
            p_ga = poisson_pmf(ga, rem_la)
            total = already_scored + gh + ga
            if total > threshold:
                p_over += p_gh * p_ga
    p_under = max(0.0, 1.0 - p_over)
    return p_over, p_under


def compute_btts(
    rem_lh: float,
    rem_la: float,
    home_score: int,
    away_score: int,
) -> float:
    """
    Return P(both teams to score), accounting for current score.

    If a team has already scored, they only need 0 more for their half
    of the condition to be satisfied.
    """
    p_home_scores = 1.0 if home_score > 0 else 1.0 - poisson_pmf(0, rem_lh)
    p_away_scores = 1.0 if away_score > 0 else 1.0 - poisson_pmf(0, rem_la)
    return p_home_scores * p_away_scores

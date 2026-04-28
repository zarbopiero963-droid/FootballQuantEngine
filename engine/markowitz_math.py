"""
Pure mathematical helpers for Markowitz mean-variance portfolio optimisation.

Extracted from markowitz_optimizer.py to keep each module under 400 LOC.
All public names are re-exported from engine.markowitz_optimizer for backward compat.
"""

from __future__ import annotations

import logging
import math
from typing import List, Tuple

from engine.markowitz_types import BetProposal, PortfolioAllocation

logger = logging.getLogger(__name__)

_PSD_EPS = 1e-8


# ---------------------------------------------------------------------------
# Pure portfolio mathematics
# ---------------------------------------------------------------------------


def portfolio_stats(
    weights: List[float],
    mu: List[float],
    cov: List[List[float]],
) -> Tuple[float, float, float]:
    """Compute (expected_return, variance, sharpe_ratio) for a weight vector."""
    num = len(weights)
    exp_ret = sum(weights[i] * mu[i] for i in range(num))
    variance = 0.0
    for row in range(num):
        for col in range(num):
            variance += weights[row] * cov[row][col] * weights[col]
    variance = max(variance, 0.0)
    std_p = math.sqrt(variance)
    sharpe = exp_ret / std_p if std_p > 1e-12 else 0.0
    return exp_ret, variance, sharpe


def sharpe_gradient(
    weights: List[float],
    mu: List[float],
    cov: List[List[float]],
) -> List[float]:
    """Analytical gradient ∇_w Sharpe = (μ_i·σ_p − μ_p·(Σw)_i/σ_p) / σ²_p."""
    num = len(weights)
    exp_ret, variance, _ = portfolio_stats(weights, mu, cov)
    std_p = math.sqrt(max(variance, 0.0))
    if std_p < 1e-12:
        return [0.0] * num
    sigma_w = [sum(cov[r][c] * weights[c] for c in range(num)) for r in range(num)]
    return [
        (mu[i] * std_p - exp_ret * (sigma_w[i] / std_p)) / variance
        for i in range(num)
    ]


def variance_gradient(
    weights: List[float],
    mu: List[float],
    cov: List[List[float]],
    gamma: float,
) -> List[float]:
    """Negative gradient of σ²_p − γ·μ_p (ascent direction for descent on f)."""
    num = len(weights)
    sigma_w = [sum(cov[r][c] * weights[c] for c in range(num)) for r in range(num)]
    return [gamma * mu[i] - 2.0 * sigma_w[i] for i in range(num)]


# ---------------------------------------------------------------------------
# Constraint helpers
# ---------------------------------------------------------------------------


def bet_correlation(
    bet_a: BetProposal,
    bet_b: BetProposal,
    same_match_rho: float,
    within_group_rho: float,
    cross_group_rho: float,
) -> float:
    """Pairwise correlation between two bets using piecewise group rules."""
    if (
        bet_a.same_match_group
        and bet_b.same_match_group
        and bet_a.same_match_group == bet_b.same_match_group
    ):
        return same_match_rho
    if bet_a.correlation_group == bet_b.correlation_group:
        return within_group_rho
    return cross_group_rho


def build_cov_matrix(
    bets: List[BetProposal],
    same_match_rho: float,
    within_group_rho: float,
    cross_group_rho: float,
) -> List[List[float]]:
    """Build the N×N covariance matrix with PSD regularisation."""
    num = len(bets)
    stds = [math.sqrt(max(b.variance, 0.0)) for b in bets]
    cov: List[List[float]] = [[0.0] * num for _ in range(num)]
    for row in range(num):
        for col in range(num):
            if row == col:
                cov[row][col] = bets[row].variance
            else:
                rho = bet_correlation(bets[row], bets[col], same_match_rho, within_group_rho, cross_group_rho)
                cov[row][col] = rho * stds[row] * stds[col]
    try:
        import numpy as np

        arr = np.array(cov)
        min_eig = float(np.linalg.eigvalsh(arr).min())
        if min_eig < _PSD_EPS:
            shift = _PSD_EPS - min_eig
            for i in range(num):
                cov[i][i] += shift
    except Exception:
        pass
    return cov


def project_weights(
    weights: List[float],
    bets: List[BetProposal],
    bankroll_fraction: float,
) -> List[float]:
    """Clip weights to [0, max_stake_fraction] and scale so sum ≤ bankroll_fraction."""
    projected = [max(0.0, min(w, bets[i].max_stake_fraction)) for i, w in enumerate(weights)]
    total = sum(projected)
    if total > bankroll_fraction and total > 1e-12:
        scale = bankroll_fraction / total
        projected = [w * scale for w in projected]
    return projected


def warm_start(
    bets: List[BetProposal],
    kelly_cap: float,
    bankroll_fraction: float,
) -> List[float]:
    """Initialise weights from capped Kelly fractions normalised to bankroll_fraction."""
    raw = [min(b.kelly_fraction, kelly_cap) for b in bets]
    total = sum(raw)
    if total < 1e-12:
        return [bankroll_fraction / len(bets)] * len(bets)
    scale = min(1.0, bankroll_fraction / total)
    return [w * scale for w in raw]


def gradient_step(
    weights: List[float],
    mu: List[float],
    cov: List[List[float]],
    gamma: float,
    learning_rate: float,
) -> List[float]:
    """One gradient step on the γ-parameterised variance–return objective."""
    grad = variance_gradient(weights, mu, cov, gamma)
    return [w + learning_rate * g for w, g in zip(weights, grad)]


def optimise_gamma(
    bets: List[BetProposal],
    mu: List[float],
    cov: List[List[float]],
    gamma: float,
    learning_rate: float,
    max_iterations: int,
    tolerance: float,
    kelly_cap: float,
    bankroll_fraction: float,
) -> List[float]:
    """Projected gradient descent on σ²_p − γ·μ_p to convergence."""
    weights = warm_start(bets, kelly_cap, bankroll_fraction)
    weights = project_weights(weights, bets, bankroll_fraction)
    for iteration in range(max_iterations):
        new_w = gradient_step(weights, mu, cov, gamma, learning_rate)
        new_w = project_weights(new_w, bets, bankroll_fraction)
        delta = max(abs(new_w[i] - weights[i]) for i in range(len(weights)))
        weights = new_w
        if delta < tolerance:
            logger.debug("γ=%.4f converged at iteration %d", gamma, iteration)
            break
    return weights


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def filter_positive_edge(bets: List[BetProposal]) -> List[BetProposal]:
    """Discard bets with non-positive expected return."""
    filtered = [b for b in bets if b.expected_return > 0.0]
    removed = len(bets) - len(filtered)
    if removed:
        logger.info("Removed %d bets with non-positive expected return.", removed)
    return filtered


def empty_allocation(method: str) -> PortfolioAllocation:
    """Return a zero PortfolioAllocation when no valid bets remain."""
    return PortfolioAllocation(
        bets=[],
        weights=[],
        expected_return=0.0,
        portfolio_variance=0.0,
        portfolio_std=0.0,
        sharpe_ratio=0.0,
        total_allocated=0.0,
        n_bets=0,
        optimisation_method=method,
    )

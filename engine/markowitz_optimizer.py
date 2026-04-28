"""
Markowitz Mean-Variance Portfolio Optimizer for correlated bet slates.

Finds weight vector w that maximises Sharpe ratio subject to:
  w_i ≥ 0  |  Σ w_i ≤ bankroll_fraction  |  w_i ≤ max_stake_fraction

Pure-Python implementation (no NumPy required for core optimisation).
For usage examples see engine.markowitz_math and the optimise_portfolio()
convenience function at the bottom of this module.
"""

from __future__ import annotations

import logging
import math
from typing import List

from engine.markowitz_math import (
    build_cov_matrix,
    empty_allocation,
    filter_positive_edge,
    optimise_gamma,
    portfolio_stats,
    project_weights,
    sharpe_gradient,
    warm_start,
)
from engine.markowitz_types import (
    BetProposal,
    EfficientFrontier,
    EfficientFrontierPoint,
    PortfolioAllocation,
)

logger = logging.getLogger(__name__)

# Re-export types for backward compatibility
__all__ = [
    "BetProposal",
    "EfficientFrontier",
    "EfficientFrontierPoint",
    "MarkowitzOptimizer",
    "PortfolioAllocation",
    "optimise_portfolio",
]


# ---------------------------------------------------------------------------
# MarkowitzOptimizer
# ---------------------------------------------------------------------------


class MarkowitzOptimizer:
    """
    Gradient-based Markowitz mean-variance portfolio optimiser for bet slates.

    Parameters
    ----------
    within_group_rho:
        Correlation between bets in the same competition round / group.
    same_match_rho:
        Correlation between bets on the same match.
    cross_group_rho:
        Baseline correlation between bets in different groups.
    kelly_cap:
        Upper bound applied to each individual Kelly fraction before warm-start.
    bankroll_fraction:
        Maximum fraction of total bankroll allocated across all bets.
    learning_rate:
        Step size for projected gradient ascent.
    max_iterations:
        Maximum gradient-ascent steps before early stopping.
    tolerance:
        Convergence threshold on the L∞ norm of the weight delta.
    """

    def __init__(
        self,
        within_group_rho: float = 0.35,
        same_match_rho: float = 0.60,
        cross_group_rho: float = 0.10,
        kelly_cap: float = 0.25,
        bankroll_fraction: float = 0.80,
        learning_rate: float = 0.001,
        max_iterations: int = 2000,
        tolerance: float = 1e-6,
    ) -> None:
        self.within_group_rho = within_group_rho
        self.same_match_rho = same_match_rho
        self.cross_group_rho = cross_group_rho
        self.kelly_cap = kelly_cap
        self.bankroll_fraction = bankroll_fraction
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def optimise(self, bets: List[BetProposal]) -> PortfolioAllocation:
        """Maximise Sharpe ratio via projected gradient ascent."""
        bets = filter_positive_edge(bets)
        if not bets:
            return empty_allocation("max_sharpe")

        mu = [b.expected_return for b in bets]
        cov = self._cov(bets)
        weights = warm_start(bets, self.kelly_cap, self.bankroll_fraction)
        weights = project_weights(weights, bets, self.bankroll_fraction)

        for iteration in range(self.max_iterations):
            grad = sharpe_gradient(weights, mu, cov)
            if max(abs(g) for g in grad) < self.tolerance:
                logger.debug("Converged at iteration %d", iteration)
                break
            weights = [w + self.learning_rate * g for w, g in zip(weights, grad)]
            weights = project_weights(weights, bets, self.bankroll_fraction)

        exp_ret, variance, sharpe = portfolio_stats(weights, mu, cov)
        return PortfolioAllocation(
            bets=bets,
            weights=weights,
            expected_return=exp_ret,
            portfolio_variance=variance,
            portfolio_std=math.sqrt(max(variance, 0.0)),
            sharpe_ratio=sharpe,
            total_allocated=sum(weights),
            n_bets=len(bets),
            optimisation_method="max_sharpe",
        )

    def min_variance(self, bets: List[BetProposal]) -> PortfolioAllocation:
        """Minimise portfolio variance (γ→0 in mean-variance objective)."""
        bets = filter_positive_edge(bets)
        if not bets:
            return empty_allocation("min_variance")

        mu = [b.expected_return for b in bets]
        cov = self._cov(bets)
        weights = self._opt_gamma(bets, mu, cov, gamma=0.001)
        exp_ret, variance, sharpe = portfolio_stats(weights, mu, cov)
        return PortfolioAllocation(
            bets=bets,
            weights=weights,
            expected_return=exp_ret,
            portfolio_variance=variance,
            portfolio_std=math.sqrt(max(variance, 0.0)),
            sharpe_ratio=sharpe,
            total_allocated=sum(weights),
            n_bets=len(bets),
            optimisation_method="min_variance",
        )

    def kelly_naive(self, bets: List[BetProposal]) -> PortfolioAllocation:
        """Naïve Kelly allocation with no correlation adjustment."""
        bets = filter_positive_edge(bets)
        if not bets:
            return empty_allocation("kelly_naive")

        mu = [b.expected_return for b in bets]
        cov = self._cov(bets)
        weights = [
            min(b.kelly_fraction, self.kelly_cap, b.max_stake_fraction) for b in bets
        ]
        total = sum(weights)
        if total > self.bankroll_fraction:
            scale = self.bankroll_fraction / total
            weights = [w * scale for w in weights]

        exp_ret, variance, sharpe = portfolio_stats(weights, mu, cov)
        return PortfolioAllocation(
            bets=bets,
            weights=weights,
            expected_return=exp_ret,
            portfolio_variance=variance,
            portfolio_std=math.sqrt(max(variance, 0.0)),
            sharpe_ratio=sharpe,
            total_allocated=sum(weights),
            n_bets=len(bets),
            optimisation_method="kelly_naive",
        )

    def efficient_frontier(
        self, bets: List[BetProposal], n_points: int = 50
    ) -> EfficientFrontier:
        """
        Trace the efficient frontier by varying risk-aversion γ logarithmically
        from 0.001 (min-variance) to 50 (high-return).
        """
        bets = filter_positive_edge(bets)
        if not bets:
            dummy = EfficientFrontierPoint(0.0, 0.0, 0.0, 0.0, [])
            return EfficientFrontier(
                bets=[], points=[], max_sharpe_point=dummy, min_variance_point=dummy
            )

        mu = [b.expected_return for b in bets]
        cov = self._cov(bets)
        log_min, log_max = math.log(0.001), math.log(50.0)
        gammas = [
            math.exp(log_min + (log_max - log_min) * i / (n_points - 1))
            for i in range(n_points)
        ]

        points: List[EfficientFrontierPoint] = []
        for gamma in gammas:
            w = self._opt_gamma(bets, mu, cov, gamma)
            exp_ret, variance, sharpe = portfolio_stats(w, mu, cov)
            points.append(
                EfficientFrontierPoint(
                    risk_aversion=gamma,
                    expected_return=exp_ret,
                    portfolio_std=math.sqrt(max(variance, 0.0)),
                    sharpe_ratio=sharpe,
                    weights=list(w),
                )
            )

        return EfficientFrontier(
            bets=bets,
            points=points,
            max_sharpe_point=max(points, key=lambda pt: pt.sharpe_ratio),
            min_variance_point=min(points, key=lambda pt: pt.portfolio_std),
        )

    # ------------------------------------------------------------------
    # Private shims — delegate to standalone functions with self params
    # ------------------------------------------------------------------

    def _cov(self, bets: List[BetProposal]) -> List[List[float]]:
        return build_cov_matrix(
            bets, self.same_match_rho, self.within_group_rho, self.cross_group_rho
        )

    def _opt_gamma(
        self,
        bets: List[BetProposal],
        mu: List[float],
        cov: List[List[float]],
        gamma: float,
    ) -> List[float]:
        return optimise_gamma(
            bets,
            mu,
            cov,
            gamma,
            self.learning_rate,
            self.max_iterations,
            self.tolerance,
            self.kelly_cap,
            self.bankroll_fraction,
        )


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def optimise_portfolio(
    bets: List[dict],
    bankroll: float = 1000.0,
) -> PortfolioAllocation:
    """
    One-shot helper: convert raw dicts to BetProposal objects, run max-Sharpe
    optimisation, and return the resulting PortfolioAllocation.

    Required dict keys: bet_id, description, odds, model_prob, correlation_group.
    Optional keys: same_match_group (str), max_stake_fraction (float, default 0.05).
    """
    if not bets:
        raise ValueError("bets list must not be empty")

    proposals: List[BetProposal] = []
    for raw in bets:
        try:
            proposals.append(
                BetProposal(
                    bet_id=str(raw["bet_id"]),
                    description=str(raw["description"]),
                    odds=float(raw["odds"]),
                    model_prob=float(raw["model_prob"]),
                    correlation_group=str(raw["correlation_group"]),
                    same_match_group=str(raw.get("same_match_group", "")),
                    max_stake_fraction=float(raw.get("max_stake_fraction", 0.05)),
                )
            )
        except KeyError as exc:
            raise ValueError(f"Missing required bet field: {exc}") from exc

    optimizer = MarkowitzOptimizer()
    allocation = optimizer.optimise(proposals)

    logger.info(
        "Optimised portfolio: %d bets, total allocated=%.1f%%, "
        "E[R]=%.2f%%, Sharpe=%.3f, bankroll=%.2f",
        allocation.n_bets,
        allocation.total_allocated * 100.0,
        allocation.expected_return * 100.0,
        allocation.sharpe_ratio,
        bankroll,
    )

    stake_amounts = [w * bankroll for w in allocation.weights]
    object.__setattr__(allocation, "__stake_amounts__", stake_amounts)  # type: ignore[call-arg]

    return allocation

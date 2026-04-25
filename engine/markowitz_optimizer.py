"""
Markowitz Mean-Variance Portfolio Optimization for Bet Sizing
=============================================================

Kelly Criterion vs Markowitz — when to use which
-------------------------------------------------
The **Kelly Criterion** answers a single-bet question: given odds *o* and model
probability *p*, what fraction of bankroll maximises long-run geometric growth?
The formula is f* = (p·o − 1) / (o − 1).  Kelly is optimal *only* when bets are
independent.  In football betting they rarely are:

* Five bets on Serie A underdogs in the same round share referee tendencies,
  weather, and team motivation signals — they are positively correlated.
* A "home win" and "over 2.5 goals" on the same Arsenal–Chelsea fixture are
  driven by the same underlying team strengths; they move together.
* Even cross-league bets share a weak positive correlation through
  broad market efficiency cycles.

When bets are correlated, naive Kelly *over-allocates* capital because it
ignores the fact that losing bets tend to cluster.  The result is higher
drawdowns and slower bankroll growth than Kelly predicts.

**Markowitz Mean-Variance Optimisation** solves this by treating the betting
slate as a *portfolio*.  Each bet has:
  - an expected return  μ_i = p_i · o_i − 1
  - a variance          σ²_i = p_i · (1 − p_i) · o_i²   (Bernoulli × payout²)

Bets are linked through a **covariance matrix** Σ where
  Σ_ij = ρ_ij · σ_i · σ_j

with ρ depending on whether the bets share the same match, same competition
round, or different groups entirely.

The optimiser finds weight vector **w** (fraction of bankroll per bet) that
**maximises the Sharpe ratio**  S = μ_p / σ_p  subject to:
  * w_i ≥ 0  (no shorting)
  * Σ w_i ≤ bankroll_fraction  (stay solvent)
  * w_i ≤ max_stake_fraction   (single-bet hard cap)

Because no external numerical libraries (NumPy, SciPy) are used, the
optimisation is implemented as **projected gradient ascent** entirely with
Python lists and the standard `math` module.  This keeps the module
self-contained and deployable in any environment.

Efficient Frontier
------------------
By varying a risk-aversion parameter γ and minimising  σ²_p − γ · μ_p  we
trace the set of Pareto-optimal portfolios — the "efficient frontier".  The
highest-Sharpe point on that curve is the default recommendation.

Usage
-----
    from engine.markowitz_optimizer import optimise_portfolio

    bets = [
        {"bet_id": "b1", "description": "Napoli Win", "odds": 2.10,
         "model_prob": 0.54, "correlation_group": "serie_a_r28"},
        {"bet_id": "b2", "description": "Lazio Win", "odds": 3.20,
         "model_prob": 0.38, "correlation_group": "serie_a_r28"},
        {"bet_id": "b3", "description": "PSG Win", "odds": 1.75,
         "model_prob": 0.62, "correlation_group": "ligue1_r31"},
    ]
    allocation = optimise_portfolio(bets, bankroll=1000.0)
    print(allocation.summary())
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import List, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class BetProposal:
    """A single bet candidate with pre-computed statistics."""

    bet_id: str
    description: str
    odds: float
    model_prob: float
    correlation_group: str
    same_match_group: str = ""
    max_stake_fraction: float = 0.05

    expected_return: float = field(init=False)
    variance: float = field(init=False)
    kelly_fraction: float = field(init=False)
    edge_pct: float = field(init=False)

    def __post_init__(self) -> None:
        if not (0.0 < self.model_prob < 1.0):
            raise ValueError(
                f"model_prob must be in (0, 1), got {self.model_prob} for {self.bet_id}"
            )
        if self.odds <= 1.0:
            raise ValueError(
                f"odds must be > 1.0 (decimal), got {self.odds} for {self.bet_id}"
            )

        # μ_i = p_i · o_i − 1
        self.expected_return = self.model_prob * self.odds - 1.0

        # σ²_i = p_i · (1 − p_i) · o_i²
        self.variance = self.model_prob * (1.0 - self.model_prob) * (self.odds**2)

        # Kelly fraction: f* = (p·o − 1) / (o − 1)
        denom = self.odds - 1.0
        raw_kelly = self.expected_return / denom if denom > 1e-12 else 0.0
        self.kelly_fraction = max(0.0, raw_kelly)

        # Edge as percentage of implied probability
        implied_prob = 1.0 / self.odds
        self.edge_pct = (self.model_prob - implied_prob) * 100.0


@dataclass
class PortfolioAllocation:
    """Result of a portfolio optimisation run."""

    bets: List[BetProposal]
    weights: List[float]
    expected_return: float
    portfolio_variance: float
    portfolio_std: float
    sharpe_ratio: float
    total_allocated: float
    n_bets: int
    optimisation_method: str

    def summary(self) -> str:
        """Return an ASCII table summarising the allocation."""
        col_w = [24, 8, 8, 8, 8]
        header = (
            f"{'Bet':<{col_w[0]}} {'Weight':>{col_w[1]}} {'Kelly':>{col_w[2]}}"
            f" {'Edge%':>{col_w[3]}} {'Odds':>{col_w[4]}}"
        )
        sep = "-" * (sum(col_w) + len(col_w) - 1)
        rows = [header, sep]
        for bet, wgt in zip(self.bets, self.weights):
            desc = bet.description[: col_w[0]]
            rows.append(
                f"{desc:<{col_w[0]}} {wgt * 100:>{col_w[1]}.2f}%"
                f" {bet.kelly_fraction * 100:>{col_w[2]}.2f}%"
                f" {bet.edge_pct:>{col_w[3]}.1f}%"
                f" {bet.odds:>{col_w[4]}.2f}"
            )
        rows.append(sep)
        rows.append(
            f"Method: {self.optimisation_method}  |  "
            f"Total allocated: {self.total_allocated * 100:.1f}%  |  "
            f"E[R]: {self.expected_return * 100:.2f}%  |  "
            f"σ: {self.portfolio_std * 100:.2f}%  |  "
            f"Sharpe: {self.sharpe_ratio:.3f}"
        )
        return "\n".join(rows)


@dataclass
class EfficientFrontierPoint:
    """A single point on the mean-variance efficient frontier."""

    risk_aversion: float
    expected_return: float
    portfolio_std: float
    sharpe_ratio: float
    weights: List[float]


@dataclass
class EfficientFrontier:
    """The complete efficient frontier for a set of bets."""

    bets: List[BetProposal]
    points: List[EfficientFrontierPoint]
    max_sharpe_point: EfficientFrontierPoint
    min_variance_point: EfficientFrontierPoint

    def describe(self) -> str:
        """Return a compact text description of the frontier."""
        lines = [
            f"Efficient Frontier — {len(self.bets)} bets, {len(self.points)} points",
            f"  Max-Sharpe  : E[R]={self.max_sharpe_point.expected_return * 100:.2f}%"
            f"  σ={self.max_sharpe_point.portfolio_std * 100:.2f}%"
            f"  Sharpe={self.max_sharpe_point.sharpe_ratio:.3f}",
            f"  Min-Variance: E[R]={self.min_variance_point.expected_return * 100:.2f}%"
            f"  σ={self.min_variance_point.portfolio_std * 100:.2f}%"
            f"  Sharpe={self.min_variance_point.sharpe_ratio:.3f}",
        ]
        lines.append(f"  {'γ':>8}  {'E[R]%':>8}  {'σ%':>8}  {'Sharpe':>8}")
        lines.append("  " + "-" * 38)
        step = max(1, len(self.points) // 10)
        for pt in self.points[::step]:
            lines.append(
                f"  {pt.risk_aversion:>8.3f}  {pt.expected_return * 100:>8.2f}"
                f"  {pt.portfolio_std * 100:>8.2f}  {pt.sharpe_ratio:>8.3f}"
            )
        return "\n".join(lines)


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
        Correlation between bets on the same match (e.g., home win + over goals).
    cross_group_rho:
        Baseline correlation between bets in different groups.
    kelly_cap:
        Upper bound applied to each individual Kelly fraction before
        using as the warm-start for optimisation.
    bankroll_fraction:
        Maximum fraction of total bankroll that can be allocated across
        all bets combined.
    learning_rate:
        Step size for projected gradient ascent.
    max_iterations:
        Maximum gradient-ascent steps before early stopping.
    tolerance:
        Convergence threshold on the L∞ norm of the gradient.
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
        """
        Maximise the Sharpe ratio of the bet portfolio via projected gradient
        ascent.  Only bets with positive expected return are considered.

        Returns
        -------
        PortfolioAllocation
            The max-Sharpe allocation.
        """
        bets = self._filter_positive_edge(bets)
        if not bets:
            return self._empty_allocation("max_sharpe")

        mu = [b.expected_return for b in bets]
        cov = self._build_cov_matrix(bets)

        weights = self._warm_start(bets)
        weights = self._project_weights(weights, bets)

        for iteration in range(self.max_iterations):
            grad = self._sharpe_gradient(weights, mu, cov)
            max_grad = max(abs(g) for g in grad)
            if max_grad < self.tolerance:
                logger.debug("Converged at iteration %d", iteration)
                break
            weights = [w + self.learning_rate * g for w, g in zip(weights, grad)]
            weights = self._project_weights(weights, bets)

        exp_ret, variance, sharpe = self._portfolio_stats(weights, mu, cov)
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
        """
        Minimise portfolio variance while maintaining a positive expected
        return.  Uses gradient descent on variance with a penalty that keeps
        expected return non-negative.

        Returns
        -------
        PortfolioAllocation
            The minimum-variance allocation.
        """
        bets = self._filter_positive_edge(bets)
        if not bets:
            return self._empty_allocation("min_variance")

        mu = [b.expected_return for b in bets]
        cov = self._build_cov_matrix(bets)

        # Use a very small risk-aversion (γ→0 favours minimum variance)
        weights = self._optimise_gamma(bets, mu, cov, gamma=0.001)

        exp_ret, variance, sharpe = self._portfolio_stats(weights, mu, cov)
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
        """
        Baseline allocation: use raw (capped) Kelly fractions with no
        correlation adjustment, then scale down proportionally if total
        exceeds bankroll_fraction.

        Returns
        -------
        PortfolioAllocation
            The naïve Kelly allocation.
        """
        bets = self._filter_positive_edge(bets)
        if not bets:
            return self._empty_allocation("kelly_naive")

        mu = [b.expected_return for b in bets]
        cov = self._build_cov_matrix(bets)

        weights = [
            min(b.kelly_fraction, self.kelly_cap, b.max_stake_fraction) for b in bets
        ]
        total = sum(weights)
        if total > self.bankroll_fraction:
            scale = self.bankroll_fraction / total
            weights = [w * scale for w in weights]

        exp_ret, variance, sharpe = self._portfolio_stats(weights, mu, cov)
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
        from 0.001 (min-variance end) to 50 (high-return end).

        Each point minimises  σ²_p − γ · μ_p  via projected gradient descent.

        Parameters
        ----------
        bets:
            Bet candidates.
        n_points:
            Number of points to sample along the frontier.

        Returns
        -------
        EfficientFrontier
        """
        bets = self._filter_positive_edge(bets)
        if not bets:
            dummy = EfficientFrontierPoint(0.0, 0.0, 0.0, 0.0, [])
            return EfficientFrontier(
                bets=[], points=[], max_sharpe_point=dummy, min_variance_point=dummy
            )

        mu = [b.expected_return for b in bets]
        cov = self._build_cov_matrix(bets)

        log_min = math.log(0.001)
        log_max = math.log(50.0)
        gammas = [
            math.exp(log_min + (log_max - log_min) * idx / (n_points - 1))
            for idx in range(n_points)
        ]

        points: List[EfficientFrontierPoint] = []
        for gamma in gammas:
            weights = self._optimise_gamma(bets, mu, cov, gamma)
            exp_ret, variance, sharpe = self._portfolio_stats(weights, mu, cov)
            points.append(
                EfficientFrontierPoint(
                    risk_aversion=gamma,
                    expected_return=exp_ret,
                    portfolio_std=math.sqrt(max(variance, 0.0)),
                    sharpe_ratio=sharpe,
                    weights=list(weights),
                )
            )

        max_sharpe_pt = max(points, key=lambda pt: pt.sharpe_ratio)
        min_variance_pt = min(points, key=lambda pt: pt.portfolio_std)

        return EfficientFrontier(
            bets=bets,
            points=points,
            max_sharpe_point=max_sharpe_pt,
            min_variance_point=min_variance_pt,
        )

    # ------------------------------------------------------------------
    # Private helpers — matrix operations
    # ------------------------------------------------------------------

    def _build_cov_matrix(self, bets: List[BetProposal]) -> List[List[float]]:
        """
        Construct the N×N covariance matrix from bet standard deviations and
        the piecewise correlation rules (same match > same group > cross group).

        Returns
        -------
        List[List[float]]
            Symmetric positive-semi-definite covariance matrix.
        """
        num = len(bets)
        stds = [math.sqrt(max(b.variance, 0.0)) for b in bets]
        cov: List[List[float]] = [[0.0] * num for _ in range(num)]

        for row in range(num):
            for col in range(num):
                if row == col:
                    cov[row][col] = bets[row].variance
                else:
                    rho = self._correlation(bets[row], bets[col])
                    cov[row][col] = rho * stds[row] * stds[col]
        return cov

    def _correlation(self, bet_a: BetProposal, bet_b: BetProposal) -> float:
        """Return the pairwise correlation between two bets."""
        if (
            bet_a.same_match_group
            and bet_b.same_match_group
            and bet_a.same_match_group == bet_b.same_match_group
        ):
            return self.same_match_rho
        if bet_a.correlation_group == bet_b.correlation_group:
            return self.within_group_rho
        return self.cross_group_rho

    def _portfolio_stats(
        self,
        weights: List[float],
        mu: List[float],
        cov: List[List[float]],
    ) -> Tuple[float, float, float]:
        """
        Compute portfolio expected return, variance, and Sharpe ratio.

        Returns
        -------
        Tuple[float, float, float]
            (expected_return, variance, sharpe_ratio)
        """
        num = len(weights)

        # μ_p = w · μ
        exp_ret = sum(weights[idx] * mu[idx] for idx in range(num))

        # σ²_p = wᵀ Σ w  (double loop)
        variance = 0.0
        for row in range(num):
            for col in range(num):
                variance += weights[row] * cov[row][col] * weights[col]
        variance = max(variance, 0.0)

        std_p = math.sqrt(variance)
        sharpe = exp_ret / std_p if std_p > 1e-12 else 0.0
        return exp_ret, variance, sharpe

    def _sharpe_gradient(
        self,
        weights: List[float],
        mu: List[float],
        cov: List[List[float]],
    ) -> List[float]:
        """
        Analytical gradient of the Sharpe ratio with respect to weights.

        ∂S/∂w_i = (μ_i · σ_p − μ_p · (∂σ_p/∂w_i)) / σ²_p
        ∂σ_p/∂w_i = (Σw)_i / σ_p

        Returns
        -------
        List[float]
            Gradient vector ∇_w S.
        """
        num = len(weights)
        exp_ret, variance, _ = self._portfolio_stats(weights, mu, cov)
        std_p = math.sqrt(max(variance, 0.0))

        if std_p < 1e-12:
            return [0.0] * num

        # (Σw)_i  — matrix-vector product
        sigma_w = [
            sum(cov[row][col] * weights[col] for col in range(num))
            for row in range(num)
        ]

        grad = [
            (mu[idx] * std_p - exp_ret * (sigma_w[idx] / std_p)) / variance
            for idx in range(num)
        ]
        return grad

    def _variance_gradient(
        self,
        weights: List[float],
        mu: List[float],
        cov: List[List[float]],
        gamma: float,
    ) -> List[float]:
        """
        Gradient of the objective  f = σ²_p − γ · μ_p  used for efficient
        frontier tracing.  We *descend* this to minimise variance for a given
        return target controlled by γ.

        ∂f/∂w_i = 2·(Σw)_i − γ·μ_i

        Returns
        -------
        List[float]
            Negative gradient (ascent direction to minimise f).
        """
        num = len(weights)
        sigma_w = [
            sum(cov[row][col] * weights[col] for col in range(num))
            for row in range(num)
        ]
        # Negative gradient of f → we use gradient *ascent* on −f in
        # _optimise_gamma, which is gradient descent on f.
        neg_grad = [gamma * mu[idx] - 2.0 * sigma_w[idx] for idx in range(num)]
        return neg_grad

    def _gradient_step(
        self,
        weights: List[float],
        mu: List[float],
        cov: List[List[float]],
        gamma: float,
    ) -> List[float]:
        """
        One projected gradient step for the γ-parameterised objective.

        Parameters
        ----------
        weights:
            Current weight vector.
        mu:
            Expected returns.
        cov:
            Covariance matrix.
        gamma:
            Risk-aversion parameter controlling return–variance tradeoff.

        Returns
        -------
        List[float]
            Updated (un-projected) weights.
        """
        grad = self._variance_gradient(weights, mu, cov, gamma)
        return [w + self.learning_rate * g for w, g in zip(weights, grad)]

    def _project_weights(
        self, weights: List[float], bets: List[BetProposal]
    ) -> List[float]:
        """
        Project weight vector back onto the feasible set:
        1. Clip each w_i to [0, bet.max_stake_fraction].
        2. Scale down proportionally if Σw_i > bankroll_fraction.

        Returns
        -------
        List[float]
            Feasible weight vector.
        """
        projected = [
            max(0.0, min(w, bets[idx].max_stake_fraction))
            for idx, w in enumerate(weights)
        ]
        total = sum(projected)
        if total > self.bankroll_fraction and total > 1e-12:
            scale = self.bankroll_fraction / total
            projected = [w * scale for w in projected]
        return projected

    # ------------------------------------------------------------------
    # Private helpers — optimisation routines
    # ------------------------------------------------------------------

    def _warm_start(self, bets: List[BetProposal]) -> List[float]:
        """
        Initialise weights from capped Kelly fractions, normalised so that
        the total allocation is proportional to bankroll_fraction.

        Returns
        -------
        List[float]
            Initial weight vector.
        """
        raw = [min(b.kelly_fraction, self.kelly_cap) for b in bets]
        total = sum(raw)
        if total < 1e-12:
            return [self.bankroll_fraction / len(bets)] * len(bets)
        scale = min(1.0, self.bankroll_fraction / total)
        return [w * scale for w in raw]

    def _optimise_gamma(
        self,
        bets: List[BetProposal],
        mu: List[float],
        cov: List[List[float]],
        gamma: float,
    ) -> List[float]:
        """
        Run projected gradient descent on  σ²_p − γ · μ_p  to convergence.

        Parameters
        ----------
        bets:
            Bet proposals (for projection constraints).
        mu:
            Expected returns.
        cov:
            Covariance matrix.
        gamma:
            Risk-aversion parameter.

        Returns
        -------
        List[float]
            Converged weight vector.
        """
        weights = self._warm_start(bets)
        weights = self._project_weights(weights, bets)

        for iteration in range(self.max_iterations):
            new_weights = self._gradient_step(weights, mu, cov, gamma)
            new_weights = self._project_weights(new_weights, bets)
            delta = max(
                abs(new_weights[idx] - weights[idx]) for idx in range(len(weights))
            )
            weights = new_weights
            if delta < self.tolerance:
                logger.debug("γ=%.4f converged at iteration %d", gamma, iteration)
                break

        return weights

    # ------------------------------------------------------------------
    # Private helpers — utilities
    # ------------------------------------------------------------------

    def _filter_positive_edge(self, bets: List[BetProposal]) -> List[BetProposal]:
        """Discard bets with non-positive expected return."""
        filtered = [b for b in bets if b.expected_return > 0.0]
        removed = len(bets) - len(filtered)
        if removed:
            logger.info("Removed %d bets with non-positive expected return.", removed)
        return filtered

    def _empty_allocation(self, method: str) -> PortfolioAllocation:
        """Return a zero allocation when no valid bets are available."""
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


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def optimise_portfolio(
    bets: List[dict],
    bankroll: float = 1000.0,
) -> PortfolioAllocation:
    """
    One-shot helper: convert raw dictionaries to BetProposal objects, run
    max-Sharpe optimisation, and return the resulting PortfolioAllocation.

    Each dict must contain:
        bet_id           (str)
        description      (str)
        odds             (float) — decimal odds, e.g. 2.50
        model_prob       (float) — model-estimated win probability in (0, 1)
        correlation_group (str)  — e.g. "serie_a_r28"

    Optional keys:
        same_match_group  (str, default "")
        max_stake_fraction (float, default 0.05)

    Parameters
    ----------
    bets:
        List of bet dictionaries.
    bankroll:
        Absolute bankroll in currency units.  Only used for logging / display
        in the summary table; all internal calculations use fractions.

    Returns
    -------
    PortfolioAllocation
        Max-Sharpe portfolio allocation.

    Example
    -------
    >>> proposals = [
    ...     {"bet_id": "b1", "description": "Napoli Win", "odds": 2.10,
    ...      "model_prob": 0.54, "correlation_group": "serie_a_r28"},
    ...     {"bet_id": "b2", "description": "PSG Win", "odds": 1.75,
    ...      "model_prob": 0.62, "correlation_group": "ligue1_r31"},
    ... ]
    >>> allocation = optimise_portfolio(proposals, bankroll=1000.0)
    >>> print(allocation.summary())
    """
    if not bets:
        raise ValueError("bets list must not be empty")

    proposals: List[BetProposal] = []
    for raw in bets:
        try:
            proposal = BetProposal(
                bet_id=str(raw["bet_id"]),
                description=str(raw["description"]),
                odds=float(raw["odds"]),
                model_prob=float(raw["model_prob"]),
                correlation_group=str(raw["correlation_group"]),
                same_match_group=str(raw.get("same_match_group", "")),
                max_stake_fraction=float(raw.get("max_stake_fraction", 0.05)),
            )
        except KeyError as exc:
            raise ValueError(f"Missing required bet field: {exc}") from exc
        proposals.append(proposal)

    optimizer = MarkowitzOptimizer()
    allocation = optimizer.optimise(proposals)

    # Annotate summary with absolute stake sizes for the caller's bankroll
    logger.info(
        "Optimised portfolio: %d bets, total allocated=%.1f%%, "
        "E[R]=%.2f%%, Sharpe=%.3f, bankroll=%.2f",
        allocation.n_bets,
        allocation.total_allocated * 100.0,
        allocation.expected_return * 100.0,
        allocation.sharpe_ratio,
        bankroll,
    )

    # Attach stake amounts as a convenience attribute without changing the
    # dataclass contract (done via __dict__ to remain dataclass-compatible).
    stake_amounts = [w * bankroll for w in allocation.weights]
    object.__setattr__(allocation, "__stake_amounts__", stake_amounts)  # type: ignore[call-arg]

    return allocation

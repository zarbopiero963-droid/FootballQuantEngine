"""
Data types for Markowitz portfolio optimisation.

Extracted from markowitz_optimizer.py to keep each module under 400 LOC.
All names are re-exported from engine.markowitz_optimizer for backward compat.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


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

        self.expected_return = self.model_prob * self.odds - 1.0
        self.variance = self.model_prob * (1.0 - self.model_prob) * (self.odds ** 2)

        denom = self.odds - 1.0
        raw_kelly = self.expected_return / denom if denom > 1e-12 else 0.0
        self.kelly_fraction = max(0.0, raw_kelly)

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

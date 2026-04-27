"""
Data types for the Gaussian Copula Engine.

Extracted from gaussian_copula.py to keep each module under 400 LOC.
All names are re-exported from engine.gaussian_copula for backward compat.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class BetLeg:
    """A single leg of a Bet Builder / Same-Game Parlay."""

    name: str
    market_odds: float
    model_prob: float

    @property
    def bookmaker_implied(self) -> float:
        """Implied probability from bookmaker decimal odds (1 / market_odds)."""
        if self.market_odds <= 0.0:
            raise ValueError(f"market_odds must be positive, got {self.market_odds}")
        return 1.0 / self.market_odds

    def __post_init__(self) -> None:
        if not (0.0 < self.model_prob < 1.0):
            raise ValueError(
                f"model_prob must be in (0, 1), got {self.model_prob!r} "
                f"for leg '{self.name}'"
            )
        if self.market_odds <= 1.0:
            raise ValueError(
                f"market_odds must be > 1.0, got {self.market_odds!r} "
                f"for leg '{self.name}'"
            )


@dataclass
class CopulaCorrelation:
    """Estimated pairwise Pearson correlation between two bet legs."""

    leg_i: int
    leg_j: int
    rho: float

    def __post_init__(self) -> None:
        if not (-1.0 <= self.rho <= 1.0):
            raise ValueError(f"rho must be in [-1, 1], got {self.rho!r}")
        if self.leg_i == self.leg_j:
            raise ValueError("leg_i and leg_j must be distinct.")


@dataclass
class CopulaResult:
    """Output of a Gaussian Copula evaluation for a multi-leg combination."""

    legs: List[BetLeg]
    correlation_matrix: List[List[float]]
    model_joint_prob: float
    book_joint_prob: float
    book_combined_odds: float
    fair_combined_odds: float
    edge_pct: float
    value_ratio: float
    n_simulations: int
    is_value: bool

    def __str__(self) -> str:
        lines = [
            "=" * 62,
            "Gaussian Copula Bet Builder Analysis",
            "=" * 62,
        ]
        for idx, leg in enumerate(self.legs):
            lines.append(
                f"  Leg {idx + 1}: {leg.name}"
                f"  |  odds={leg.market_odds:.2f}"
                f"  |  model_p={leg.model_prob:.4f}"
            )
        lines += [
            "-" * 62,
            f"  Book joint prob   : {self.book_joint_prob:.6f}",
            f"  Model joint prob  : {self.model_joint_prob:.6f}",
            f"  Book combined odds: {self.book_combined_odds:.2f}",
            f"  Fair combined odds: {self.fair_combined_odds:.2f}",
            f"  Edge              : {self.edge_pct:+.2f}%",
            f"  Value ratio       : {self.value_ratio:.3f}x",
            f"  Simulations       : {self.n_simulations:,}",
            f"  Is value bet      : {'YES' if self.is_value else 'NO'}",
            "=" * 62,
        ]
        return "\n".join(lines)

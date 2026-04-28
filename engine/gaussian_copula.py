"""
Gaussian Copula Engine for Bet Builder / Same-Game Parlay mispricing detection.

Bookmakers price multi-leg combinations by multiplying independent marginal
probabilities.  This engine estimates the true joint probability via a Gaussian
copula (Monte Carlo with Cholesky-correlated normals) and flags combinations
where the bookmaker price implies a significantly lower joint probability than
the model.

Mathematical primitives live in engine.copula_math; data types in engine.copula_types.
"""

from __future__ import annotations

import itertools
import logging
from typing import List, Optional

from engine.copula_math import (
    _cholesky,
    _make_positive_definite,
    build_default_corr_matrix,
    simulate_joint_prob,
    uniform_corr_matrix,
    validate_corr_matrix,
)
from engine.copula_types import BetLeg, CopulaCorrelation, CopulaResult

__all__ = [
    "BetLeg",
    "CopulaCorrelation",
    "CopulaResult",
    "GaussianCopulaEngine",
    "evaluate_bet_builder",
]

logger = logging.getLogger(__name__)


class GaussianCopulaEngine:
    """Gaussian copula engine for Bet Builder / Same-Game Parlay pricing.

    Parameters
    ----------
    n_simulations:
        Monte Carlo paths.  50 000 gives SE ≈ 0.2 % for typical 5 % joint probs.
    default_correlation:
        Fallback Pearson ρ when no lookup entry exists.
    edge_threshold_pct:
        Minimum edge (%) for CopulaResult.is_value to be True.
    seed:
        Optional RNG seed for reproducible tests.
    """

    def __init__(
        self,
        n_simulations: int = 50_000,
        default_correlation: float = 0.10,
        edge_threshold_pct: float = 5.0,
        seed: Optional[int] = None,
    ) -> None:
        if n_simulations < 1000:
            raise ValueError("n_simulations must be at least 1 000.")
        if not (-1.0 < default_correlation < 1.0):
            raise ValueError("default_correlation must be strictly in (-1, 1).")
        if edge_threshold_pct < 0.0:
            raise ValueError("edge_threshold_pct must be non-negative.")

        self.n_simulations = n_simulations
        self.default_correlation = default_correlation
        self.edge_threshold_pct = edge_threshold_pct
        import random

        self._rng = random.Random(seed)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def evaluate(
        self,
        legs: List[BetLeg],
        correlation_matrix: Optional[List[List[float]]] = None,
        event_types: Optional[List[str]] = None,
    ) -> CopulaResult:
        """Evaluate a multi-leg Bet Builder using the Gaussian copula.

        Parameters
        ----------
        legs:
            At least 2 BetLeg entries.
        correlation_matrix:
            Optional explicit n×n matrix.  When absent, built from event_types.
        event_types:
            Event-type strings (same length as legs) for default correlation lookup.
        """
        n = len(legs)
        if n < 2:
            raise ValueError("Need at least 2 legs for a Bet Builder evaluation.")

        if correlation_matrix is not None:
            corr = validate_corr_matrix(correlation_matrix, n)
        elif event_types is not None:
            if len(event_types) != n:
                raise ValueError(
                    f"event_types length ({len(event_types)}) must match legs length ({n})."
                )
            corr = build_default_corr_matrix(event_types, self.default_correlation)
        else:
            corr = uniform_corr_matrix(n, self.default_correlation)

        lower = _cholesky(_make_positive_definite(corr))
        probs = [leg.model_prob for leg in legs]
        model_joint = simulate_joint_prob(probs, lower, self.n_simulations, self._rng)

        book_joint = 1.0
        for leg in legs:
            book_joint *= leg.bookmaker_implied

        if model_joint <= 0.0:
            logger.warning(
                "model_joint_prob rounded to zero — increase n_simulations or "
                "check that marginal probabilities are not too small."
            )
            model_joint = 1e-10

        fair_odds = 1.0 / model_joint
        book_odds = 1.0 / book_joint
        edge = (model_joint - book_joint) / book_joint * 100.0
        vratio = model_joint / book_joint

        result = CopulaResult(
            legs=legs,
            correlation_matrix=corr,
            model_joint_prob=model_joint,
            book_joint_prob=book_joint,
            book_combined_odds=book_odds,
            fair_combined_odds=fair_odds,
            edge_pct=edge,
            value_ratio=vratio,
            n_simulations=self.n_simulations,
            is_value=(edge > self.edge_threshold_pct),
        )
        logger.debug(
            "Evaluated %d-leg parlay: book_odds=%.2f fair_odds=%.2f edge=%.2f%%",
            n,
            book_odds,
            fair_odds,
            edge,
        )
        return result

    def find_value_parlays(
        self,
        available_legs: List[BetLeg],
        max_legs: int = 4,
        event_types: Optional[List[str]] = None,
    ) -> List[CopulaResult]:
        """Screen all 2-to-max_legs combinations for positive-value parlays."""
        if max_legs < 2:
            raise ValueError("max_legs must be at least 2.")
        n_pool = len(available_legs)
        if n_pool < 2:
            raise ValueError("Need at least 2 available legs to form a parlay.")

        value_results: List[CopulaResult] = []
        total_combos = 0

        for size in range(2, min(max_legs, n_pool) + 1):
            for indices in itertools.combinations(range(n_pool), size):
                total_combos += 1
                combo_legs = [available_legs[i] for i in indices]
                combo_types: Optional[List[str]] = (
                    [event_types[i] for i in indices]
                    if event_types is not None
                    else None
                )
                try:
                    result = self.evaluate(legs=combo_legs, event_types=combo_types)
                    if result.is_value:
                        value_results.append(result)
                except Exception as exc:
                    logger.warning(
                        "Skipping combination %s: %s",
                        [leg.name for leg in combo_legs],
                        exc,
                    )

        logger.info(
            "Screened %d combinations, found %d value parlays.",
            total_combos,
            len(value_results),
        )
        value_results.sort(key=lambda r: r.edge_pct, reverse=True)
        return value_results


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def evaluate_bet_builder(
    legs: List[dict],
    correlation_matrix: Optional[List[List[float]]] = None,
    n_simulations: int = 50_000,
) -> CopulaResult:
    """One-shot Bet Builder evaluation.

    Each leg dict must have: name (str), market_odds (float), model_prob (float).
    Optional key: event_type (str) for default correlation lookup.
    """
    bet_legs: List[BetLeg] = []
    event_types: List[str] = []

    for idx, leg_dict in enumerate(legs):
        missing = [
            k for k in ("name", "market_odds", "model_prob") if k not in leg_dict
        ]
        if missing:
            raise ValueError(f"Leg {idx} is missing required keys: {missing!r}")
        bet_legs.append(
            BetLeg(
                name=str(leg_dict["name"]),
                market_odds=float(leg_dict["market_odds"]),
                model_prob=float(leg_dict["model_prob"]),
            )
        )
        event_types.append(str(leg_dict.get("event_type", "")))

    engine = GaussianCopulaEngine(n_simulations=n_simulations)
    if correlation_matrix is not None:
        return engine.evaluate(legs=bet_legs, correlation_matrix=correlation_matrix)
    return engine.evaluate(legs=bet_legs, event_types=event_types)

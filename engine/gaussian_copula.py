"""
Gaussian Copula Engine for Bet Builder / Same-Game Parlay Mispricing Detection
===============================================================================

Bookmakers typically price Bet Builder / Same-Game Parlay markets by multiplying
the implied marginal probabilities of each leg, implicitly assuming the outcomes
are statistically independent.  In reality, football match events are correlated:
a home win is negatively correlated with both teams scoring (BTTS), while BTTS
is strongly positively correlated with over-goals markets.

This module exploits that structural mispricing by estimating the **true** joint
probability of a multi-leg combination via a Gaussian copula:

    1. Each leg's marginal probability is modelled independently (supplied by
       the caller, e.g. from a Poisson or xG model).
    2. A correlation matrix Σ encodes pairwise dependencies between event types.
    3. We simulate correlated uniform random variables via:
           Z  ~  N(0, I_n)            (independent standard normals)
           Y  =  L @ Z                (L = Cholesky factor of Σ)
           U_i = Φ(Y_i)              (map back to [0,1] via normal CDF)
           leg_i fires  ⟺  U_i < p_i
    4. The empirical fraction of simulations where ALL legs fire is the copula
       joint probability estimate.

When the bookmaker's independence product gives combined odds of 15.00 but the
copula model yields a fair price of 6.00, there is substantial positive expected
value — the combination is systematically underpriced by the bookmaker.

Mathematical primitives are implemented without scipy / numpy:
  - Normal CDF: Abramowitz & Stegun (1964) rational approximation, |ε| < 7.5e-8.
  - Inverse normal CDF: Peter Acklam's algorithm.
  - Cholesky decomposition: pure Python, O(n³).
  - Correlated normals: stdlib random.gauss for marginals, Cholesky for structure.

References
----------
  Abramowitz, M. and Stegun, I. A. (1964). Handbook of Mathematical Functions.
  Acklam, P. (2003). An algorithm for computing the inverse normal cumulative
      distribution function. Unpublished manuscript.
  Sklar, A. (1959). Fonctions de répartition à n dimensions et leurs marges.
"""

from __future__ import annotations

import itertools
import logging
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

__all__ = [
    "BetLeg",
    "CopulaCorrelation",
    "CopulaResult",
    "GaussianCopulaEngine",
    "evaluate_bet_builder",
]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default pairwise correlations for common football event-type pairs.
# Sourced from empirical studies of Premier League / Bundesliga seasons.
# ---------------------------------------------------------------------------
_DEFAULT_CORRELATIONS: Dict[Tuple[str, str], float] = {
    ("home_win", "over_goals"): -0.25,  # home wins are often low-scoring
    ("home_win", "btts"): -0.30,
    ("away_win", "over_goals"): 0.15,
    ("draw", "over_goals"): 0.10,
    ("draw", "btts"): 0.35,
    ("over_goals", "btts"): 0.60,  # strong positive link
    ("home_win", "home_scorer"): 0.55,
    ("home_win", "away_scorer"): -0.40,
    ("cards_over", "over_goals"): 0.20,
    ("cards_over", "draw"): 0.15,
}


# ===========================================================================
# Pure-Python mathematical primitives
# ===========================================================================


def _normal_pdf(x: float) -> float:
    """Standard normal probability density function φ(x)."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _normal_cdf(x: float) -> float:
    """Standard normal CDF Φ(x) via Abramowitz & Stegun (1964) approximation.

    The rational approximation has absolute error < 7.5e-8 for all real x.

    Parameters
    ----------
    x:
        Argument of the CDF.

    Returns
    -------
    float
        Probability in (0, 1).
    """
    if x >= 0.0:
        t = 1.0 / (1.0 + 0.2316419 * x)
        poly = t * (
            0.319381530
            + t
            * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429)))
        )
        return 1.0 - _normal_pdf(x) * poly
    else:
        return 1.0 - _normal_cdf(-x)


def _normal_ppf(p: float) -> float:
    """Inverse standard normal CDF Φ⁻¹(p) — Peter Acklam's algorithm.

    Splits the domain into three regions for numerical stability:
      - Low tail  : p < 0.02425
      - Central   : 0.02425 ≤ p ≤ 0.97575
      - High tail : p > 0.97575

    Absolute error is less than 1.15e-9 over (0, 1).

    Parameters
    ----------
    p:
        Probability in (0, 1).

    Returns
    -------
    float
        Quantile z such that Φ(z) = p.

    Raises
    ------
    ValueError
        If p is not strictly in (0, 1).
    """
    if not (0.0 < p < 1.0):
        raise ValueError(f"_normal_ppf requires p in (0, 1), got {p!r}")

    # Acklam coefficient arrays
    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    ]

    p_low = 0.02425
    p_high = 1.0 - p_low

    if p < p_low:
        # Lower tail
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    elif p <= p_high:
        # Central region
        q = p - 0.5
        r = q * q
        return (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
            * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
        )
    else:
        # Upper tail: use symmetry
        return -_normal_ppf(1.0 - p)


def _cholesky(matrix: List[List[float]]) -> List[List[float]]:
    """Lower-triangular Cholesky factor L such that matrix = L @ L^T.

    The input must be a real, symmetric positive-definite matrix.  A small
    diagonal jitter is added automatically via :func:`_make_positive_definite`
    before calling this function.

    Parameters
    ----------
    matrix:
        n×n symmetric positive-definite matrix as a list-of-lists.

    Returns
    -------
    List[List[float]]
        Lower triangular matrix L.

    Raises
    ------
    ValueError
        If the matrix is not square or a diagonal element becomes non-positive
        during factorisation (indicating non-positive-definiteness).
    """
    n = len(matrix)
    for row in matrix:
        if len(row) != n:
            raise ValueError("Cholesky requires a square matrix.")

    # Allocate L as n×n zeros
    lower = [[0.0] * n for _ in range(n)]

    for i in range(n):
        for j in range(i + 1):
            s = matrix[i][j]
            for k in range(j):
                s -= lower[i][k] * lower[j][k]
            if i == j:
                if s <= 0.0:
                    raise ValueError(
                        f"Matrix is not positive-definite: "
                        f"diagonal element [{i}][{i}] = {s}"
                    )
                lower[i][j] = math.sqrt(s)
            else:
                lower[i][j] = s / lower[j][j]

    return lower


def _make_positive_definite(
    matrix: List[List[float]],
    jitter: float = 1e-8,
) -> List[List[float]]:
    """Return a copy of *matrix* with small diagonal jitter added if needed.

    This ensures Cholesky decomposition does not fail on near-singular
    correlation matrices (e.g. due to floating-point accumulation or
    user-supplied near-dependent legs).

    Parameters
    ----------
    matrix:
        n×n symmetric matrix.
    jitter:
        Value added to each diagonal element.

    Returns
    -------
    List[List[float]]
        Modified copy of the matrix.
    """
    n = len(matrix)
    result = [row[:] for row in matrix]
    for i in range(n):
        result[i][i] += jitter
    return result


# ===========================================================================
# Dataclasses
# ===========================================================================


@dataclass
class BetLeg:
    """A single leg of a Bet Builder / Same-Game Parlay.

    Attributes
    ----------
    name:
        Human-readable description, e.g. "Man Utd to win".
    market_odds:
        Bookmaker decimal odds for this outcome, e.g. 2.50.
    model_prob:
        Our estimated marginal probability from an xG or Poisson model.
    """

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
    """Estimated pairwise Pearson correlation between two bet legs.

    Attributes
    ----------
    leg_i:
        Index of the first leg in the legs list.
    leg_j:
        Index of the second leg in the legs list.
    rho:
        Pearson correlation coefficient in [-1, +1].
    """

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
    """Output of a Gaussian Copula evaluation for a multi-leg combination.

    Attributes
    ----------
    legs:
        The bet legs that were evaluated.
    correlation_matrix:
        The n×n correlation matrix used in the simulation.
    model_joint_prob:
        Joint probability estimated by the Gaussian copula Monte Carlo.
    book_joint_prob:
        Bookmaker's implicit joint probability (product of implied marginals).
    book_combined_odds:
        1 / book_joint_prob — what the bookmaker effectively charges.
    fair_combined_odds:
        1 / model_joint_prob — our fair price.
    edge_pct:
        (model_joint_prob - book_joint_prob) / book_joint_prob × 100.
        Positive means we believe the bet is underpriced by the bookmaker.
    value_ratio:
        model_joint_prob / book_joint_prob.  >1 means positive expected value.
    n_simulations:
        Number of Monte Carlo paths used for estimation.
    is_value:
        True when edge_pct exceeds the engine's threshold.
    """

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


# ===========================================================================
# Gaussian Copula Engine
# ===========================================================================


class GaussianCopulaEngine:
    """Gaussian copula engine for Bet Builder / Same-Game Parlay pricing.

    The engine exploits the independence assumption made by most bookmakers
    when pricing multi-leg combinations.  Given model marginal probabilities
    and a pairwise correlation structure, it estimates the true joint
    probability via Monte Carlo simulation under the Gaussian copula.

    Parameters
    ----------
    n_simulations:
        Number of Monte Carlo paths for joint probability estimation.
        50 000 paths gives standard error ≈ sqrt(p(1-p)/N) ≈ 0.2 % for
        typical 5 % joint probability values.
    default_correlation:
        Fallback Pearson ρ used when no entry exists in
        :data:`_DEFAULT_CORRELATIONS` and no matrix is supplied.
    edge_threshold_pct:
        Minimum edge (in percent) for :attr:`CopulaResult.is_value` to be True.
    seed:
        Optional integer seed for the internal :class:`random.Random` instance,
        useful for reproducible tests.
    """

    def __init__(
        self,
        n_simulations: int = 50_000,
        default_correlation: float = 0.10,
        edge_threshold_pct: float = 5.0,
        seed: Optional[int] = None,
    ) -> None:
        if n_simulations < 1000:
            raise ValueError(
                "n_simulations must be at least 1 000 for reliable estimates."
            )
        if not (-1.0 < default_correlation < 1.0):
            raise ValueError("default_correlation must be strictly in (-1, 1).")
        if edge_threshold_pct < 0.0:
            raise ValueError("edge_threshold_pct must be non-negative.")

        self.n_simulations = n_simulations
        self.default_correlation = default_correlation
        self.edge_threshold_pct = edge_threshold_pct
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
            Bet legs to evaluate.  Must contain at least 2 entries.
        correlation_matrix:
            Optional explicit n×n correlation matrix.  If omitted the engine
            builds one from :data:`_DEFAULT_CORRELATIONS` using *event_types*.
        event_types:
            List of event-type strings (same length as *legs*) used to look up
            default correlations, e.g. ``["home_win", "btts", "over_goals"]``.
            Required when *correlation_matrix* is None.

        Returns
        -------
        CopulaResult
        """
        n = len(legs)
        if n < 2:
            raise ValueError("Need at least 2 legs for a Bet Builder evaluation.")

        # Build or validate the correlation matrix
        if correlation_matrix is not None:
            corr = self._validate_corr_matrix(correlation_matrix, n)
        elif event_types is not None:
            if len(event_types) != n:
                raise ValueError(
                    f"event_types length ({len(event_types)}) must match "
                    f"legs length ({n})."
                )
            corr = self._build_default_corr_matrix(event_types)
        else:
            # All-default-correlation matrix
            corr = self._uniform_corr_matrix(n, self.default_correlation)

        # Ensure positive-definiteness before Cholesky
        corr_pd = _make_positive_definite(corr)
        lower = _cholesky(corr_pd)

        # Extract marginal model probabilities
        probs = [leg.model_prob for leg in legs]

        # Monte Carlo joint probability under Gaussian copula
        model_joint = self._simulate_joint_prob(probs, lower)

        # Bookmaker independence product
        book_joint = 1.0
        for leg in legs:
            book_joint *= leg.bookmaker_implied

        # Guard against degenerate cases
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
        """Screen all 2-to-max_legs combinations for positive-value parlays.

        Parameters
        ----------
        available_legs:
            Pool of BetLeg objects to combine.
        max_legs:
            Maximum combination size to test (inclusive).
        event_types:
            Optional list of event-type strings parallel to *available_legs*.

        Returns
        -------
        List[CopulaResult]
            All combinations flagged as value, sorted by edge_pct descending.
        """
        if max_legs < 2:
            raise ValueError("max_legs must be at least 2.")

        n_pool = len(available_legs)
        if n_pool < 2:
            raise ValueError("Need at least 2 available legs to form a parlay.")

        max_combo_size = min(max_legs, n_pool)
        value_results: List[CopulaResult] = []
        total_combos = 0

        for size in range(2, max_combo_size + 1):
            for indices in itertools.combinations(range(n_pool), size):
                total_combos += 1
                combo_legs = [available_legs[idx] for idx in indices]

                # Build event_types subset if provided
                combo_types: Optional[List[str]] = None
                if event_types is not None:
                    combo_types = [event_types[idx] for idx in indices]

                try:
                    result = self.evaluate(
                        legs=combo_legs,
                        event_types=combo_types,
                    )
                    if result.is_value:
                        value_results.append(result)
                except Exception as exc:
                    logger.warning(
                        "Skipping combination %s due to error: %s",
                        [leg.name for leg in combo_legs],
                        exc,
                    )

        logger.info(
            "Screened %d combinations, found %d value parlays.",
            total_combos,
            len(value_results),
        )

        value_results.sort(key=lambda res: res.edge_pct, reverse=True)
        return value_results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _simulate_joint_prob(
        self,
        probs: List[float],
        lower: List[List[float]],
    ) -> float:
        """Monte Carlo estimate of joint probability under the Gaussian copula.

        Algorithm
        ---------
        For each simulation path:
          1. Draw n independent standard normals z_i using random.gauss.
          2. Compute correlated normals y = L @ z  (L is the Cholesky factor).
          3. Map back to uniform margins: u_i = Φ(y_i).
          4. All legs fire when u_i < p_i for all i simultaneously.

        Parameters
        ----------
        probs:
            Marginal model probabilities for each leg.
        lower:
            Lower-triangular Cholesky factor of the correlation matrix.

        Returns
        -------
        float
            Empirical fraction of paths where all legs fire.
        """
        n = len(probs)
        hits = 0

        for _ in range(self.n_simulations):
            # Step 1: independent standard normals
            z = [self._rng.gauss(0.0, 1.0) for _ in range(n)]

            # Step 2: correlate via Cholesky — y = L @ z
            y = [0.0] * n
            for i in range(n):
                s = 0.0
                for k in range(i + 1):
                    s += lower[i][k] * z[k]
                y[i] = s

            # Step 3 & 4: map to uniform, check all legs fire
            all_fire = True
            for i in range(n):
                u = _normal_cdf(y[i])
                if u >= probs[i]:
                    all_fire = False
                    break

            if all_fire:
                hits += 1

        return hits / self.n_simulations

    def _build_default_corr_matrix(
        self,
        event_types: List[str],
    ) -> List[List[float]]:
        """Build an n×n correlation matrix from :data:`_DEFAULT_CORRELATIONS`.

        Parameters
        ----------
        event_types:
            List of event-type strings corresponding to each leg.

        Returns
        -------
        List[List[float]]
            Symmetric correlation matrix with ones on the diagonal.
        """
        n = len(event_types)
        corr = self._uniform_corr_matrix(n, 0.0)  # start with identity

        for i in range(n):
            for j in range(i + 1, n):
                ti, tj = event_types[i], event_types[j]
                rho = (
                    _DEFAULT_CORRELATIONS.get((ti, tj))
                    or _DEFAULT_CORRELATIONS.get((tj, ti))
                    or self.default_correlation
                )
                corr[i][j] = rho
                corr[j][i] = rho

        return corr

    @staticmethod
    def _uniform_corr_matrix(
        n: int,
        rho: float,
    ) -> List[List[float]]:
        """Return an n×n matrix with 1.0 on the diagonal and *rho* elsewhere.

        Parameters
        ----------
        n:
            Matrix dimension.
        rho:
            Off-diagonal correlation value.

        Returns
        -------
        List[List[float]]
        """
        matrix = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                matrix[i][j] = 1.0 if i == j else rho
        return matrix

    @staticmethod
    def _validate_corr_matrix(
        matrix: List[List[float]],
        expected_n: int,
    ) -> List[List[float]]:
        """Validate shape and diagonal of a user-supplied correlation matrix.

        Parameters
        ----------
        matrix:
            Candidate correlation matrix.
        expected_n:
            Required dimension.

        Returns
        -------
        List[List[float]]
            The validated matrix (unchanged).

        Raises
        ------
        ValueError
            If shape or diagonal constraints are violated.
        """
        if len(matrix) != expected_n:
            raise ValueError(
                f"correlation_matrix must be {expected_n}×{expected_n}, "
                f"got {len(matrix)} rows."
            )
        for i, row in enumerate(matrix):
            if len(row) != expected_n:
                raise ValueError(
                    f"Row {i} of correlation_matrix has {len(row)} elements, "
                    f"expected {expected_n}."
                )
            if abs(matrix[i][i] - 1.0) > 1e-9:
                raise ValueError(
                    f"Diagonal element [{i}][{i}] = {matrix[i][i]}, expected 1.0."
                )
        return matrix


# ===========================================================================
# Module-level convenience function
# ===========================================================================


def evaluate_bet_builder(
    legs: List[dict],
    correlation_matrix: Optional[List[List[float]]] = None,
    n_simulations: int = 50_000,
) -> CopulaResult:
    """One-shot Bet Builder evaluation using the Gaussian Copula Engine.

    Converts plain dicts to :class:`BetLeg` objects and delegates to
    :class:`GaussianCopulaEngine`.

    Parameters
    ----------
    legs:
        List of dicts, each with keys:

        - ``"name"`` (str) — human-readable description
        - ``"market_odds"`` (float) — bookmaker decimal odds
        - ``"model_prob"`` (float) — model marginal probability
        - ``"event_type"`` (str, optional) — event type for default correlations

    correlation_matrix:
        Optional explicit n×n correlation matrix.  When omitted, event types
        from the leg dicts are used to look up :data:`_DEFAULT_CORRELATIONS`.

    n_simulations:
        Monte Carlo path count, default 50 000.

    Returns
    -------
    CopulaResult

    Examples
    --------
    >>> result = evaluate_bet_builder(
    ...     legs=[
    ...         {"name": "Home win", "market_odds": 2.10, "model_prob": 0.52,
    ...          "event_type": "home_win"},
    ...         {"name": "BTTS Yes", "market_odds": 1.85, "model_prob": 0.57,
    ...          "event_type": "btts"},
    ...         {"name": "Over 2.5", "market_odds": 1.90, "model_prob": 0.55,
    ...          "event_type": "over_goals"},
    ...     ],
    ...     n_simulations=50_000,
    ... )
    >>> print(result)
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

    # If no explicit matrix, try default lookup; fall back to default rho
    if correlation_matrix is not None:
        return engine.evaluate(
            legs=bet_legs,
            correlation_matrix=correlation_matrix,
        )
    else:
        return engine.evaluate(
            legs=bet_legs,
            event_types=event_types,
        )

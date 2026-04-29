"""
Pure-Python mathematical primitives for the Gaussian, Clayton and Gumbel
Copula engines.

Extracted from gaussian_copula.py to keep each module under 400 LOC.
Implements normal CDF/PPF (Abramowitz & Stegun / Acklam), Cholesky
decomposition, correlation matrix helpers, and Monte Carlo joint
probability simulation — all without external dependencies.

Copula families
---------------
Gaussian  — zero tail dependence; suitable for independent/moderate correlations.
Clayton   — lower tail dependence; models joint crashes (e.g. accumulators that
            all lose together when the favourite underperforms).
            θ ∈ (0, ∞), lower-tail coefficient λ_L = 2^(−1/θ).
Gumbel    — upper tail dependence; models joint wins on correlated favourites.
            θ ∈ [1, ∞), upper-tail coefficient λ_U = 2 − 2^(1/θ).
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Tuple


def _normal_pdf(x: float) -> float:
    """Standard normal probability density function φ(x)."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _normal_cdf(x: float) -> float:
    """Standard normal CDF Φ(x) via Abramowitz & Stegun (1964) approximation.
    Absolute error < 7.5e-8 for all real x.
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
    Absolute error < 1.15e-9 over (0, 1).
    """
    if not (0.0 < p < 1.0):
        raise ValueError(f"_normal_ppf requires p in (0, 1), got {p!r}")

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
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    elif p <= p_high:
        q = p - 0.5
        r = q * q
        return (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
            * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
        )
    else:
        return -_normal_ppf(1.0 - p)


def _cholesky(matrix: List[List[float]]) -> List[List[float]]:
    """Lower-triangular Cholesky factor L such that matrix = L @ L^T."""
    n = len(matrix)
    for row in matrix:
        if len(row) != n:
            raise ValueError("Cholesky requires a square matrix.")

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
    """Return a copy of *matrix* with small diagonal jitter to ensure PSD."""
    n = len(matrix)
    result = [row[:] for row in matrix]
    for i in range(n):
        result[i][i] += jitter
    return result


# ---------------------------------------------------------------------------
# Correlation matrix helpers
# ---------------------------------------------------------------------------

# Default pairwise Pearson correlations between football event types.
DEFAULT_CORRELATIONS: Dict[Tuple[str, str], float] = {
    ("home_win", "over_goals"): -0.25,
    ("home_win", "btts"): -0.30,
    ("away_win", "over_goals"): 0.15,
    ("draw", "over_goals"): 0.10,
    ("draw", "btts"): 0.35,
    ("over_goals", "btts"): 0.60,
    ("home_win", "home_scorer"): 0.55,
    ("home_win", "away_scorer"): -0.40,
    ("cards_over", "over_goals"): 0.20,
    ("cards_over", "draw"): 0.15,
}


def uniform_corr_matrix(n: int, rho: float) -> List[List[float]]:
    """Return an n×n matrix with 1.0 on the diagonal and *rho* off-diagonal."""
    matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            matrix[i][j] = 1.0 if i == j else rho
    return matrix


def validate_corr_matrix(
    matrix: List[List[float]], expected_n: int
) -> List[List[float]]:
    """Validate shape and diagonal of a user-supplied correlation matrix."""
    if len(matrix) != expected_n:
        raise ValueError(
            f"correlation_matrix must be {expected_n}×{expected_n}, got {len(matrix)} rows."
        )
    for i, row in enumerate(matrix):
        if len(row) != expected_n:
            raise ValueError(
                f"Row {i} of correlation_matrix has {len(row)} elements, expected {expected_n}."
            )
        if abs(matrix[i][i] - 1.0) > 1e-9:
            raise ValueError(
                f"Diagonal element [{i}][{i}] = {matrix[i][i]}, expected 1.0."
            )
    return matrix


def build_default_corr_matrix(
    event_types: List[str],
    default_correlation: float = 0.10,
) -> List[List[float]]:
    """Build an n×n correlation matrix from DEFAULT_CORRELATIONS lookup."""
    n = len(event_types)
    corr = uniform_corr_matrix(n, 0.0)
    for i in range(n):
        for j in range(i + 1, n):
            ti, tj = event_types[i], event_types[j]
            rho = (
                DEFAULT_CORRELATIONS.get((ti, tj))
                or DEFAULT_CORRELATIONS.get((tj, ti))
                or default_correlation
            )
            corr[i][j] = rho
            corr[j][i] = rho
    return corr


# ---------------------------------------------------------------------------
# Monte Carlo simulation
# ---------------------------------------------------------------------------


def simulate_joint_prob(
    probs: List[float],
    lower: List[List[float]],
    n_simulations: int,
    rng: Optional[random.Random] = None,
) -> float:
    """Monte Carlo estimate of joint probability under the Gaussian copula.

    For each path: draw n independent N(0,1), correlate via Cholesky L,
    map to uniform margins with Φ, count when all u_i < p_i.
    """
    if rng is None:
        rng = random.Random()
    n = len(probs)
    hits = 0
    for _ in range(n_simulations):
        z = [rng.gauss(0.0, 1.0) for _ in range(n)]
        all_fire = True
        for i in range(n):
            y_i = sum(lower[i][k] * z[k] for k in range(i + 1))
            if _normal_cdf(y_i) >= probs[i]:
                all_fire = False
                break
        if all_fire:
            hits += 1
    return hits / n_simulations


# ---------------------------------------------------------------------------
# Archimedean copula simulation helpers
# ---------------------------------------------------------------------------


def _gamma_sample(alpha: float, rng: random.Random) -> float:
    """Sample from Gamma(alpha, 1) using Marsaglia-Tsang method (alpha >= 1)
    or the squeeze/rejection method for alpha < 1.

    Used internally by Clayton and Gumbel copula samplers.
    """
    if alpha < 1.0:
        # Boost: Gamma(alpha) = Gamma(alpha+1) * U^(1/alpha)
        return _gamma_sample(alpha + 1.0, rng) * (rng.random() ** (1.0 / alpha))

    # Marsaglia-Tsang (2000): d = alpha - 1/3, c = 1/sqrt(9d)
    d = alpha - 1.0 / 3.0
    c = 1.0 / math.sqrt(9.0 * d)
    while True:
        x = rng.gauss(0.0, 1.0)
        v = (1.0 + c * x) ** 3
        if v <= 0.0:
            continue
        u = rng.random()
        x2 = x * x
        if u < 1.0 - 0.0331 * (x2 * x2):
            return d * v
        if math.log(u) < 0.5 * x2 + d * (1.0 - v + math.log(v)):
            return d * v


def _stable_sample(alpha: float, rng: random.Random) -> float:
    """Sample from a totally-skewed positive α-stable distribution S(α,1) via
    the Chambers-Mallows-Stuck (1976) algorithm, as described in Hofert (2011).

    Used by the Gumbel copula sampler (Marshall-Olkin representation).
    α must be in (0, 1).
    """
    u = rng.random() * math.pi  # U ~ Uniform(0, π)
    e = -math.log(max(rng.random(), 1e-300))  # E ~ Exp(1), guard log(0)
    # Hofert (2011) Algorithm 3: use sin((1-α)u), NOT cos((1-α)u).
    # cos would be negative for u > π/(2(1-α)) when α < 0.5 (θ > 2),
    # producing complex numbers when raised to a fractional power.
    sin_alpha_u = math.sin(alpha * u)
    sin_1m_alpha_u = math.sin((1.0 - alpha) * u)
    sin_u = math.sin(u)
    if sin_u < 1e-300:
        return 1.0  # u ≈ 0 or π: stable → 1 (independence fallback)
    return (sin_alpha_u / (sin_u ** (1.0 / alpha))) * (
        (sin_1m_alpha_u / e) ** ((1.0 - alpha) / alpha)
    )


def simulate_joint_prob_clayton(
    probs: List[float],
    theta: float,
    n_simulations: int,
    rng: Optional[random.Random] = None,
) -> float:
    """Monte Carlo estimate of joint probability under the Clayton copula.

    Uses the Marshall-Olkin representation:
      1. Sample V ~ Gamma(1/θ, 1).
      2. Sample E_i ~ Exp(1) independently for i = 1, …, n.
      3. U_i = (1 + E_i / V)^(−1/θ).
    Lower-tail dependence coefficient: λ_L = 2^(−1/θ).

    Parameters
    ----------
    probs:
        Marginal probabilities (transformed to uniform margins before counting).
    theta:
        Clayton parameter θ > 0.  θ → 0 gives independence; θ → ∞ comonotonicity.
    """
    if theta <= 0.0:
        raise ValueError(f"Clayton theta must be > 0, got {theta}")
    if rng is None:
        rng = random.Random()
    n = len(probs)
    inv_theta = 1.0 / theta
    hits = 0
    for _ in range(n_simulations):
        v = _gamma_sample(inv_theta, rng)
        if v == 0.0:
            # Gamma(1/θ) underflows to 0 for large θ; V→0 ⟹ U_i→0 for all i.
            # Count as hit only when all p_i > 0 — consistent with normal path
            # where u_i=0.0 >= p_i=0.0 would correctly reject a zero-prob event.
            if all(p > 0.0 for p in probs):
                hits += 1
            continue
        all_fire = True
        for i in range(n):
            e_i = -math.log(rng.random())
            u_i = (1.0 + e_i / v) ** (-inv_theta)
            if u_i >= probs[i]:
                all_fire = False
                break
        if all_fire:
            hits += 1
    return hits / n_simulations


def simulate_joint_prob_gumbel(
    probs: List[float],
    theta: float,
    n_simulations: int,
    rng: Optional[random.Random] = None,
) -> float:
    """Monte Carlo estimate of joint probability under the Gumbel copula.

    Uses the Marshall-Olkin representation via a positive α-stable variable:
      1. Sample S ~ TotallySkewed-PositiveStable(1/θ) [Chambers-Mallows-Stuck].
      2. Sample E_i ~ Exp(1) independently for i = 1, …, n.
      3. U_i = exp(−(E_i / S)^(1/θ)).
    Upper-tail dependence coefficient: λ_U = 2 − 2^(1/θ).

    Parameters
    ----------
    probs:
        Marginal probabilities.
    theta:
        Gumbel parameter θ ≥ 1.  θ = 1 gives independence; θ → ∞ comonotonicity.
    """
    if theta < 1.0:
        raise ValueError(f"Gumbel theta must be >= 1, got {theta}")
    if rng is None:
        rng = random.Random()
    n = len(probs)
    alpha = 1.0 / theta  # stable index ∈ (0, 1]
    hits = 0
    for _ in range(n_simulations):
        if alpha >= 1.0:
            # θ = 1 → independence
            s = 1.0
        else:
            s = _stable_sample(alpha, rng)
        all_fire = True
        for i in range(n):
            e_i = -math.log(rng.random())
            u_i = math.exp(-((e_i / s) ** alpha))
            if u_i >= probs[i]:
                all_fire = False
                break
        if all_fire:
            hits += 1
    return hits / n_simulations

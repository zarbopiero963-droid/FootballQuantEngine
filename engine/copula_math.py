"""
Pure-Python mathematical primitives for the Gaussian Copula Engine.

Extracted from gaussian_copula.py to keep each module under 400 LOC.
Implements normal CDF/PPF (Abramowitz & Stegun / Acklam) and Cholesky
decomposition without any external dependencies.
"""

from __future__ import annotations

import math
from typing import List


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

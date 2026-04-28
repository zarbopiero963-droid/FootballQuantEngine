"""
Archimedean Copulas: Clayton and Gumbel families.

Gaussian copulas assume symmetric tail dependence.  Real football multi-leg
bets often exhibit asymmetric tail dependence:

* **Lower tail** (both legs fail together in a disrupted match) is captured
  well by the **Clayton copula**.
* **Upper tail** (both legs succeed together in an attacking display) is
  captured well by the **Gumbel copula**.

Using the appropriate copula family reduces mispricing error on same-game
parlays compared with the Gaussian assumption.

Joint probability is computed via the closed-form Archimedean CDF:

  Clayton:  C(u_1,...,u_n; θ) = (Σ u_i^{−θ} − n + 1)^{−1/θ}
  Gumbel:   C(u_1,...,u_n; θ) = exp(−(Σ (−ln u_i)^θ)^{1/θ})

These are exact (no sampling variance) and O(n) in the number of legs.

References
----------
  Nelsen, R. B. (2006). An Introduction to Copulas. Springer.
  Joe, H. (2014). Dependence Modeling with Copulas. CRC Press.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import List

logger = logging.getLogger(__name__)


@dataclass
class ArchimedeanResult:
    """Joint probability from an Archimedean copula."""

    joint_probability: float
    copula_family: str
    theta: float
    leg_probs: List[float]

    @property
    def independence_product(self) -> float:
        p = 1.0
        for prob in self.leg_probs:
            p *= prob
        return p

    @property
    def copula_vs_independence_ratio(self) -> float:
        ind = self.independence_product
        if ind <= 0:
            return 1.0
        return self.joint_probability / ind


class ClaytonCopula:
    """
    Clayton copula with parameter θ > 0.

    Exhibits lower-tail dependence: both legs fail together more often
    than independence predicts when θ > 0.

    Tail dependence:
        Lower: λ_L = 2^{−1/θ}
        Upper: λ_U = 0

    Joint CDF (exact, closed form):
        C(u_1,...,u_n; θ) = (Σ u_i^{−θ} − n + 1)^{−1/θ}
    """

    def __init__(self, theta: float = 1.0):
        if theta <= 0:
            raise ValueError(f"Clayton theta must be > 0, got {theta}")
        self.theta = float(theta)

    @property
    def lower_tail_dependence(self) -> float:
        """λ_L = 2^{−1/θ}"""
        return 2.0 ** (-1.0 / self.theta)

    def joint_probability(self, leg_probs: List[float]) -> ArchimedeanResult:
        """
        Compute P(all legs fire) exactly via Clayton copula CDF.

        Parameters
        ----------
        leg_probs : marginal probabilities in (0, 1) for each parlay leg
        """
        if not leg_probs:
            return ArchimedeanResult(1.0, "Clayton", self.theta, [])

        probs = [max(1e-9, min(1 - 1e-9, float(p))) for p in leg_probs]
        n = len(probs)

        # C(u_1,...,u_n; θ) = (Σ u_i^{-θ} - n + 1)^{-1/θ}
        inner = sum(p ** (-self.theta) for p in probs) - n + 1
        if inner <= 0:
            jp = 0.0
        else:
            jp = inner ** (-1.0 / self.theta)

        return ArchimedeanResult(
            joint_probability=min(1.0, max(0.0, jp)),
            copula_family="Clayton",
            theta=self.theta,
            leg_probs=probs,
        )

    # Backward-compat alias
    def simulate_joint_probability(self, leg_probs: List[float]) -> ArchimedeanResult:
        return self.joint_probability(leg_probs)


class GumbelCopula:
    """
    Gumbel copula with parameter θ ≥ 1.

    Exhibits upper-tail dependence: both legs succeed together more often
    than independence predicts when θ > 1.

    Tail dependence:
        Lower: λ_L = 0
        Upper: λ_U = 2 − 2^{1/θ}

    Joint CDF (exact, closed form):
        C(u_1,...,u_n; θ) = exp(−(Σ (−ln u_i)^θ)^{1/θ})
    """

    def __init__(self, theta: float = 2.0):
        if theta < 1.0:
            raise ValueError(f"Gumbel theta must be >= 1.0, got {theta}")
        self.theta = float(theta)

    @property
    def upper_tail_dependence(self) -> float:
        """λ_U = 2 − 2^{1/θ}"""
        return 2.0 - 2.0 ** (1.0 / self.theta)

    def joint_probability(self, leg_probs: List[float]) -> ArchimedeanResult:
        """
        Compute P(all legs fire) exactly via Gumbel copula CDF.

        Parameters
        ----------
        leg_probs : marginal probabilities in (0, 1) for each parlay leg
        """
        if not leg_probs:
            return ArchimedeanResult(1.0, "Gumbel", self.theta, [])

        probs = [max(1e-9, min(1 - 1e-9, float(p))) for p in leg_probs]

        # C(u_1,...,u_n; θ) = exp(−(Σ (−ln u_i)^θ)^{1/θ})
        inner = sum((-math.log(p)) ** self.theta for p in probs)
        jp = math.exp(-(inner ** (1.0 / self.theta)))

        return ArchimedeanResult(
            joint_probability=min(1.0, max(0.0, jp)),
            copula_family="Gumbel",
            theta=self.theta,
            leg_probs=probs,
        )

    # Backward-compat alias
    def simulate_joint_probability(self, leg_probs: List[float]) -> ArchimedeanResult:
        return self.joint_probability(leg_probs)


def make_copula(
    family: str,
    theta: float,
) -> "ClaytonCopula | GumbelCopula":
    """
    Factory returning the requested copula instance.

    Parameters
    ----------
    family : "clayton" or "gumbel" (case-insensitive)
    theta  : copula parameter (Clayton > 0; Gumbel >= 1)
    """
    f = family.lower().strip()
    if f == "clayton":
        return ClaytonCopula(theta=theta)
    if f == "gumbel":
        return GumbelCopula(theta=theta)
    raise ValueError(f"Unknown copula family '{family}'. Choose 'clayton' or 'gumbel'.")

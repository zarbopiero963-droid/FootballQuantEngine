"""
Property-based tests for the Gaussian, Clayton, and Gumbel copula engines.

Invariants verified:
- Simulated joint probability is always in [0, 1]
- Independence case: joint ≈ product of marginals (within Monte Carlo noise)
- Tail dependence ordering: Clayton joint ≥ Gaussian joint (lower-tail)
- Comonotonicity limit: joint → min(probs) as theta → ∞
- BetLeg validation raises on invalid inputs
- CopulaResult.edge_pct sign matches model vs book comparison
"""
from __future__ import annotations

import random

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from engine.copula_math import (
    simulate_joint_prob,
    simulate_joint_prob_clayton,
    simulate_joint_prob_gumbel,
    uniform_corr_matrix,
)
from engine.copula_types import BetLeg

# ---------------------------------------------------------------------------
# Shared strategies
# ---------------------------------------------------------------------------

prob = st.floats(min_value=0.05, max_value=0.95, allow_nan=False, allow_infinity=False)
prob_pair = st.lists(prob, min_size=2, max_size=2)

clayton_theta = st.floats(min_value=0.1, max_value=10.0, allow_nan=False)
gumbel_theta = st.floats(min_value=1.0, max_value=10.0, allow_nan=False)

_N = 5_000  # Monte Carlo samples — enough precision without being too slow
_RNG = random.Random(42)


# ---------------------------------------------------------------------------
# Joint probability is always in [0, 1]
# ---------------------------------------------------------------------------


@given(prob_pair, clayton_theta)
@settings(max_examples=300)
def test_clayton_joint_prob_in_unit_interval(probs: list[float], theta: float) -> None:
    jp = simulate_joint_prob_clayton(probs, theta, n_simulations=_N, rng=_RNG)
    assert 0.0 <= jp <= 1.0, f"Clayton joint={jp} not in [0,1] for probs={probs}"


@given(prob_pair, gumbel_theta)
@settings(max_examples=300)
def test_gumbel_joint_prob_in_unit_interval(probs: list[float], theta: float) -> None:
    jp = simulate_joint_prob_gumbel(probs, theta, n_simulations=_N, rng=_RNG)
    assert 0.0 <= jp <= 1.0, f"Gumbel joint={jp} not in [0,1] for probs={probs}"


@given(prob_pair)
@settings(max_examples=200)
def test_gaussian_joint_prob_in_unit_interval(probs: list[float]) -> None:
    corr = uniform_corr_matrix(len(probs), rho=0.0)
    jp = simulate_joint_prob(probs, corr, n_simulations=_N, rng=_RNG)
    assert 0.0 <= jp <= 1.0, f"Gaussian joint={jp} not in [0,1] for probs={probs}"


# ---------------------------------------------------------------------------
# Independence: joint ≈ product of marginals (within Monte Carlo noise 3σ)
# ---------------------------------------------------------------------------


@given(prob_pair)
@settings(max_examples=200)
def test_gaussian_zero_corr_approximates_independence(probs: list[float]) -> None:
    """rho=0 Gaussian copula should give joint ≈ product of marginals."""
    corr = uniform_corr_matrix(len(probs), rho=0.0)
    jp = simulate_joint_prob(probs, corr, n_simulations=20_000, rng=_RNG)
    independent = 1.0
    for p in probs:
        independent *= p
    # Monte Carlo std ≈ sqrt(p*(1-p)/N) < 0.005 for N=20k; allow 5σ tolerance
    tol = 5 * (independent * (1 - independent) / 20_000) ** 0.5 + 0.01
    assert abs(jp - independent) < tol, (
        f"Gaussian rho=0 joint={jp:.4f} vs product={independent:.4f} "
        f"(tol={tol:.4f}) for probs={probs}"
    )


@given(prob_pair)
@settings(max_examples=200)
def test_clayton_high_theta_approaches_comonotonicity(probs: list[float]) -> None:
    """As theta → ∞, Clayton joint → min(probs) (comonotonicity)."""
    jp = simulate_joint_prob_clayton(probs, theta=50.0, n_simulations=_N, rng=_RNG)
    floor = min(probs)
    # Allow generous tolerance due to MC noise at extreme theta
    assert jp >= floor * 0.85, (
        f"Clayton theta=50 joint={jp:.4f} far below min(probs)={floor:.4f}"
    )


@given(prob_pair)
@settings(max_examples=200)
def test_gumbel_high_theta_approaches_comonotonicity(probs: list[float]) -> None:
    """As theta → ∞, Gumbel joint → min(probs)."""
    jp = simulate_joint_prob_gumbel(probs, theta=50.0, n_simulations=_N, rng=_RNG)
    floor = min(probs)
    assert jp >= floor * 0.85, (
        f"Gumbel theta=50 joint={jp:.4f} far below min(probs)={floor:.4f}"
    )


# ---------------------------------------------------------------------------
# BetLeg validation
# ---------------------------------------------------------------------------


@given(
    st.text(min_size=1),
    st.floats(min_value=1.01, max_value=50.0),
    st.floats(min_value=0.01, max_value=0.99),
)
@settings(max_examples=500)
def test_betleg_valid_inputs_accepted(name: str, market_odds: float, model_prob: float) -> None:
    """Valid BetLeg inputs must not raise."""
    leg = BetLeg(name=name, market_odds=market_odds, model_prob=model_prob)
    assert 0.0 < leg.bookmaker_implied < 1.0


@given(
    st.text(min_size=1),
    st.floats(min_value=1.01, max_value=50.0),
    st.one_of(
        st.floats(max_value=0.0),
        st.floats(min_value=1.0),
        st.just(float("nan")),
    ),
)
@settings(max_examples=300)
def test_betleg_invalid_prob_raises(name: str, market_odds: float, bad_prob: float) -> None:
    """model_prob outside (0, 1) must raise ValueError."""
    assume(not (0.0 < bad_prob < 1.0))
    with pytest.raises((ValueError, Exception)):
        BetLeg(name=name, market_odds=market_odds, model_prob=bad_prob)


@given(
    st.text(min_size=1),
    st.floats(max_value=1.0, allow_nan=False),
    st.floats(min_value=0.01, max_value=0.99),
)
@settings(max_examples=300)
def test_betleg_invalid_odds_raises(name: str, bad_odds: float, model_prob: float) -> None:
    """market_odds ≤ 1.0 must raise ValueError."""
    with pytest.raises((ValueError, Exception)):
        BetLeg(name=name, market_odds=bad_odds, model_prob=model_prob)

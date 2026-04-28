"""
Property-based tests for Kelly criterion, Expected Value, Sharpe ratio,
and EdgeDetector using Hypothesis.

These tests verify mathematical invariants that must hold for ALL valid
inputs — not just the happy-path examples in unit tests.
"""
from __future__ import annotations

import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ranking.match_ranker import expected_value, kelly_fraction, sharpe_ratio
from strategies.edge_detector import EdgeDetector

# ---------------------------------------------------------------------------
# Shared strategies
# ---------------------------------------------------------------------------

# Valid probability domain (open interval to avoid floating-point extremes)
prob = st.floats(min_value=0.01, max_value=0.99, allow_nan=False, allow_infinity=False)

# Valid decimal odds (must be > 1.0 to make a bet meaningful)
odds = st.floats(min_value=1.01, max_value=50.0, allow_nan=False, allow_infinity=False)


# ---------------------------------------------------------------------------
# Kelly criterion — 3 invariants
# ---------------------------------------------------------------------------


@given(prob, odds)
@settings(max_examples=2000)
def test_kelly_bounded(p: float, o: float) -> None:
    """Kelly fraction is always in [0, 1]."""
    kf = kelly_fraction(p, o)
    assert 0.0 <= kf <= 1.0, f"kelly={kf} out of [0,1] for p={p}, odds={o}"


@given(prob, odds)
@settings(max_examples=2000)
def test_kelly_zero_when_negative_ev(p: float, o: float) -> None:
    """Kelly must be 0 whenever EV ≤ 0 (never bet on negative-EV outcomes)."""
    ev = expected_value(p, o)
    kf = kelly_fraction(p, o)
    if ev <= 0:
        assert kf == 0.0, f"kelly={kf} but EV={ev} ≤ 0 for p={p}, odds={o}"


@given(prob, st.floats(min_value=1.01, max_value=20.0, allow_nan=False))
@settings(max_examples=1000)
def test_kelly_monotone_in_probability(o: float, _unused: float) -> None:
    """For fixed odds, higher probability → higher Kelly fraction."""
    p1, p2 = 0.3, 0.6
    kf1 = kelly_fraction(p1, o)
    kf2 = kelly_fraction(p2, o)
    assert kf1 <= kf2, (
        f"kelly not monotone: kelly({p1}, {o})={kf1} > kelly({p2}, {o})={kf2}"
    )


# ---------------------------------------------------------------------------
# Expected Value — 3 invariants
# ---------------------------------------------------------------------------


@given(prob, odds)
@settings(max_examples=2000)
def test_ev_positive_iff_p_exceeds_implied(p: float, o: float) -> None:
    """EV > 0 iff model probability exceeds bookmaker's implied probability."""
    implied = 1.0 / o
    ev = expected_value(p, o)
    if p > implied:
        assert ev > 0, f"EV={ev} not positive when p={p} > implied={implied}"
    elif p < implied:
        assert ev < 0, f"EV={ev} not negative when p={p} < implied={implied}"


@given(st.floats(min_value=1.01, max_value=30.0, allow_nan=False))
@settings(max_examples=1000)
def test_ev_monotone_in_probability(o: float) -> None:
    """For fixed odds, EV strictly increases with probability."""
    p_lo, p_hi = 0.2, 0.8
    assert expected_value(p_lo, o) < expected_value(p_hi, o), (
        f"EV not monotone in p at odds={o}"
    )


@given(prob, odds)
@settings(max_examples=2000)
def test_ev_is_finite(p: float, o: float) -> None:
    """EV is always a finite float — no NaN, no inf."""
    ev = expected_value(p, o)
    assert math.isfinite(ev), f"EV={ev} is not finite for p={p}, odds={o}"


# ---------------------------------------------------------------------------
# Sharpe ratio — 2 invariants
# ---------------------------------------------------------------------------


@given(prob, st.floats(min_value=-5.0, max_value=5.0, allow_nan=False))
@settings(max_examples=1000)
def test_sharpe_finite(p: float, ev: float) -> None:
    """Sharpe is always a finite float."""
    sr = sharpe_ratio(p, ev)
    assert math.isfinite(sr), f"Sharpe={sr} is not finite for p={p}, EV={ev}"


@given(prob, st.floats(min_value=0.01, max_value=5.0, allow_nan=False))
@settings(max_examples=1000)
def test_sharpe_positive_for_positive_ev(p: float, ev: float) -> None:
    """Sharpe inherits the sign of EV."""
    sr = sharpe_ratio(p, ev)
    assert sr > 0, f"Sharpe={sr} not positive for EV={ev}>0, p={p}"


# ---------------------------------------------------------------------------
# EdgeDetector — 2 invariants
# ---------------------------------------------------------------------------


@given(prob, odds)
@settings(max_examples=2000)
def test_edge_sign_matches_value(p: float, o: float) -> None:
    """Edge is positive iff model probability exceeds bookmaker implied prob."""
    det = EdgeDetector()
    edge = det.calculate_edge(p, o)
    implied = 1.0 / o
    if p > implied:
        assert edge > 0, f"edge={edge} not positive when p={p} > implied={implied}"
    else:
        assert edge <= 0, f"edge={edge} not non-positive when p={p} <= implied={implied}"


@given(prob, odds)
@settings(max_examples=2000)
def test_is_value_bet_consistent_with_edge(p: float, o: float) -> None:
    """is_value_bet() must agree with calculate_edge() > EDGE_THRESHOLD."""
    from config.constants import EDGE_THRESHOLD

    det = EdgeDetector()
    edge = det.calculate_edge(p, o)
    is_value = det.is_value_bet(p, o)
    expected = edge > EDGE_THRESHOLD
    assert is_value == expected, (
        f"is_value_bet={is_value} but edge={edge}, threshold={EDGE_THRESHOLD}"
    )

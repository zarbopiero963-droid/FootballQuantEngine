"""
Property-based tests for MarketTools and probability normalisation.

Invariants:
- normalize_implied_probs_1x2 always sums to exactly 1.0
- All three normalised probabilities are in (0, 1)
- fair_odds is the exact reciprocal of probability
- edge() sign matches model_prob vs 1/bookmaker_odds comparison
"""
from __future__ import annotations

import math

from hypothesis import given, settings
from hypothesis import strategies as st

from quant.services.market_tools import MarketTools

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

positive_odds = st.floats(min_value=1.01, max_value=50.0, allow_nan=False, allow_infinity=False)
prob = st.floats(min_value=0.01, max_value=0.99, allow_nan=False, allow_infinity=False)

odds_triple = st.fixed_dictionaries(
    {"home": positive_odds, "draw": positive_odds, "away": positive_odds}
)

_mt = MarketTools()


# ---------------------------------------------------------------------------
# normalize_implied_probs_1x2
# ---------------------------------------------------------------------------


@given(odds_triple)
@settings(max_examples=2000)
def test_normalized_probs_sum_to_one(odds: dict) -> None:
    """After overround removal, 1x2 probabilities must sum to exactly 1."""
    probs = _mt.normalize_implied_probs_1x2(odds)
    total = probs["home_win"] + probs["draw"] + probs["away_win"]
    assert abs(total - 1.0) < 1e-10, (
        f"Probabilities sum to {total} (delta={total - 1.0}) for odds={odds}"
    )


@given(odds_triple)
@settings(max_examples=2000)
def test_normalized_probs_all_in_unit_interval(odds: dict) -> None:
    """Each normalised probability is strictly in (0, 1)."""
    probs = _mt.normalize_implied_probs_1x2(odds)
    for market, p in probs.items():
        assert 0.0 < p < 1.0, (
            f"prob[{market}]={p} not in (0, 1) for odds={odds}"
        )


@given(odds_triple)
@settings(max_examples=1000)
def test_higher_odds_implies_lower_probability(odds: dict) -> None:
    """The outcome with the highest odds must have the lowest probability."""
    probs = _mt.normalize_implied_probs_1x2(odds)
    max_odds_key = max(odds, key=lambda k: odds[k])
    prob_map = {"home": "home_win", "draw": "draw", "away": "away_win"}
    min_prob_key = min(probs, key=lambda k: probs[k])
    assert min_prob_key == prob_map[max_odds_key], (
        f"Highest odds on '{max_odds_key}' but lowest prob on '{min_prob_key}'"
    )


# ---------------------------------------------------------------------------
# fair_odds
# ---------------------------------------------------------------------------


@given(prob)
@settings(max_examples=2000)
def test_fair_odds_is_reciprocal_of_probability(p: float) -> None:
    """fair_odds(p) == 1/p for any valid probability."""
    fo = _mt.fair_odds(p)
    assert abs(fo - 1.0 / p) < 1e-9, f"fair_odds({p})={fo}, expected {1.0/p}"


@given(prob)
@settings(max_examples=1000)
def test_fair_odds_monotone_decreasing(p: float) -> None:
    """Higher probability → lower fair odds (inverse relationship)."""
    p2 = min(p + 0.1, 0.99)
    assert _mt.fair_odds(p) >= _mt.fair_odds(p2), (
        f"fair_odds not decreasing: fair_odds({p})={_mt.fair_odds(p)} "
        f"< fair_odds({p2})={_mt.fair_odds(p2)}"
    )


# ---------------------------------------------------------------------------
# edge()
# ---------------------------------------------------------------------------


@given(prob, positive_odds)
@settings(max_examples=2000)
def test_edge_sign_matches_value_comparison(p: float, o: float) -> None:
    """edge > 0 iff model_prob * odds > 1 (positive expected profit)."""
    edge = _mt.edge(p, o)
    profitable = p * o > 1.0
    if profitable:
        assert edge > 0, f"edge={edge} not positive when p*odds={p*o:.4f}>1"
    else:
        assert edge <= 0, f"edge={edge} not non-positive when p*odds={p*o:.4f}<=1"


@given(prob, positive_odds)
@settings(max_examples=1000)
def test_edge_is_finite(p: float, o: float) -> None:
    """edge() always returns a finite float."""
    assert math.isfinite(_mt.edge(p, o))

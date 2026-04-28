"""
Financial invariants for the quant engine.

These are absolute correctness guarantees — if any of these fail, the engine
is producing mathematically unsound output that would cause real financial loss.
Unlike property tests (random inputs via Hypothesis), these use a fixed set of
representative scenarios that cover boundary conditions and typical market data.
"""
from __future__ import annotations

import math

import pytest

from engine.copula_math import simulate_joint_prob_clayton, simulate_joint_prob_gumbel
from quant.services.agreement_engine import AgreementEngine
from quant.services.confidence_engine import QuantConfidenceEngine
from quant.services.market_tools import MarketTools
from quant.services.no_bet_filter import QuantNoBetFilter
from ranking.match_ranker import expected_value, kelly_fraction, sharpe_ratio
from strategies.edge_detector import EdgeDetector
from strategies.no_bet_filter import NoBetFilter

_mt = MarketTools()
_det = EdgeDetector()
_conf = QuantConfidenceEngine()
_agree = AgreementEngine()
_filter = QuantNoBetFilter()


# ---------------------------------------------------------------------------
# I.  Probability sum invariant
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "odds",
    [
        {"home": 2.10, "draw": 3.40, "away": 3.20},  # typical 1x2
        {"home": 1.50, "draw": 4.00, "away": 6.00},  # heavy favourite
        {"home": 3.00, "draw": 3.00, "away": 2.50},  # away favourite
        {"home": 1.01, "draw": 50.0, "away": 50.0},  # extreme favourite
        {"home": 10.0, "draw": 10.0, "away": 1.05},  # extreme away
    ],
)
def test_implied_probs_sum_to_one(odds: dict) -> None:
    probs = _mt.normalize_implied_probs_1x2(odds)
    total = sum(probs.values())
    assert abs(total - 1.0) < 1e-10, f"probs sum={total} for odds={odds}"


# ---------------------------------------------------------------------------
# II.  Kelly ∈ [0, 1] — never bet more than 100% of bankroll
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "p, o",
    [
        (0.55, 2.10),   # clear value bet
        (0.33, 3.00),   # fair price — edge ≈ 0
        (0.20, 3.00),   # negative edge
        (0.99, 1.01),   # near certainty, tiny odds
        (0.01, 50.0),   # long shot
        (0.50, 1.90),   # vigorish erodes edge
    ],
)
def test_kelly_in_unit_interval(p: float, o: float) -> None:
    kf = kelly_fraction(p, o)
    assert 0.0 <= kf <= 1.0, f"Kelly={kf} outside [0,1] for p={p}, odds={o}"


# ---------------------------------------------------------------------------
# III.  Kelly = 0 whenever EV ≤ 0
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "p, o",
    [
        (0.30, 2.00),  # implied=0.50 > 0.30 → negative EV
        (0.33, 2.50),  # implied=0.40 > 0.33 → negative EV
        (0.10, 1.50),  # implied=0.67 >> 0.10
        (0.49, 2.10),  # implied=0.476 < 0.49 → positive EV
    ],
)
def test_kelly_zero_iff_no_edge(p: float, o: float) -> None:
    ev = expected_value(p, o)
    kf = kelly_fraction(p, o)
    if ev <= 0:
        assert kf == 0.0, f"Kelly={kf} should be 0 when EV={ev:.4f}≤0"
    else:
        assert kf > 0.0, f"Kelly={kf} should be >0 when EV={ev:.4f}>0"


# ---------------------------------------------------------------------------
# IV.  EV monotone in probability (same odds)
# ---------------------------------------------------------------------------


def test_ev_strictly_increasing_in_probability() -> None:
    odds = 2.50
    probs = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    evs = [expected_value(p, odds) for p in probs]
    for i in range(len(evs) - 1):
        assert evs[i] < evs[i + 1], (
            f"EV not monotone: EV({probs[i]})={evs[i]:.4f} "
            f">= EV({probs[i+1]})={evs[i+1]:.4f}"
        )


# ---------------------------------------------------------------------------
# V.  No NaN / Inf in any quant output
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "p, o",
    [
        (0.01, 1.01),
        (0.99, 50.0),
        (0.50, 2.00),
        (0.33, 3.03),
    ],
)
def test_no_nan_or_inf_in_quant_outputs(p: float, o: float) -> None:
    for name, val in [
        ("ev", expected_value(p, o)),
        ("kelly", kelly_fraction(p, o)),
        ("sharpe", sharpe_ratio(p, expected_value(p, o))),
        ("edge", _det.calculate_edge(p, o)),
        ("fair_odds", _mt.fair_odds(p)),
        ("market_edge", _mt.edge(p, o)),
    ]:
        assert math.isfinite(val), f"{name}={val} is not finite for p={p}, odds={o}"


# ---------------------------------------------------------------------------
# VI.  Confidence score ∈ [0, 1]
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "prob, edge, agreement, xg",
    [
        (0.60, 0.10, 0.75, 0.80),
        (0.50, 0.00, 0.50, 0.00),
        (0.99, 1.00, 1.00, 1.00),   # extreme high
        (0.01, 0.00, 0.00, 0.00),   # extreme low
    ],
)
def test_confidence_score_bounded(prob, edge, agreement, xg) -> None:
    score = _conf.score(prob, edge, agreement, xg)
    assert 0.0 <= score <= 1.0, f"confidence={score} out of [0,1]"


# ---------------------------------------------------------------------------
# VII.  Agreement score ∈ [0, 1]
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "probs_list",
    [
        [{"home_win": 0.50, "draw": 0.25, "away_win": 0.25}],                    # single model
        [{"home_win": 0.50, "draw": 0.25, "away_win": 0.25}] * 3,               # perfect agreement
        [
            {"home_win": 0.30, "draw": 0.35, "away_win": 0.35},
            {"home_win": 0.70, "draw": 0.15, "away_win": 0.15},
        ],                                                                         # max disagreement
        [],                                                                        # empty list
    ],
)
def test_agreement_score_bounded(probs_list) -> None:
    score = _agree.three_way_agreement(probs_list)
    assert 0.0 <= score <= 1.0, f"agreement={score} out of [0,1]"


# ---------------------------------------------------------------------------
# VIII.  NoBetFilter decision space is exhaustive
# ---------------------------------------------------------------------------


def test_no_bet_filter_only_returns_valid_decisions() -> None:
    valid = {"BET", "WATCHLIST", "NO_BET"}
    scenarios = [
        (0.60, 0.10, 0.75, 0.70),
        (0.54, 0.05, 0.65, 0.60),
        (0.52, 0.03, 0.62, 0.57),
        (0.40, 0.01, 0.50, 0.40),
        (0.99, 1.00, 1.00, 1.00),
        (0.01, 0.00, 0.00, 0.00),
    ]
    for p, e, c, a in scenarios:
        decision = _filter.decide(p, e, c, a)
        assert decision in valid, f"Unknown decision '{decision}' for {(p,e,c,a)}"


# ---------------------------------------------------------------------------
# IX.  Copula joint probability ≤ min(marginals)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "probs",
    [
        [0.70, 0.60],
        [0.50, 0.50],
        [0.30, 0.80],
        [0.20, 0.20],
    ],
)
def test_joint_prob_cannot_exceed_smallest_marginal(probs: list) -> None:
    """P(A ∩ B) ≤ min(P(A), P(B)) — fundamental probability bound."""
    import random

    rng = random.Random(0)
    floor = min(probs)

    jp_c = simulate_joint_prob_clayton(probs, theta=2.0, n_simulations=10_000, rng=rng)
    assert jp_c <= floor + 0.03, (
        f"Clayton joint={jp_c:.4f} > min(probs)={floor:.4f} (with MC tolerance)"
    )

    jp_g = simulate_joint_prob_gumbel(probs, theta=2.0, n_simulations=10_000, rng=rng)
    assert jp_g <= floor + 0.03, (
        f"Gumbel joint={jp_g:.4f} > min(probs)={floor:.4f} (with MC tolerance)"
    )


# ---------------------------------------------------------------------------
# X.  Edge detector: value bet only when model exceeds implied probability
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "p, o, expect_value",
    [
        (0.55, 2.10, True),   # implied=0.476 < 0.55 → value
        (0.45, 2.10, False),  # implied=0.476 > 0.45 → no value
        (0.50, 2.00, False),  # fair price, edge=0
        (0.60, 1.50, False),  # implied=0.667 > 0.60 → no value despite high prob
        (0.70, 1.80, True),   # implied=0.556 < 0.70 → value
    ],
)
def test_is_value_bet_matches_edge_sign(p: float, o: float, expect_value: bool) -> None:
    from config.constants import EDGE_THRESHOLD

    edge = _det.calculate_edge(p, o)
    is_value = _det.is_value_bet(p, o)
    assert is_value == (edge > EDGE_THRESHOLD), (
        f"is_value_bet={is_value} but edge={edge:.4f}, threshold={EDGE_THRESHOLD}"
    )

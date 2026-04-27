"""
Unit tests for the quant core modules.

All tests run without API keys — no external services are called.
"""

import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ranking.match_ranker import (
    kelly_fraction,
    expected_value,
    sharpe_ratio,
    MatchRanker,
)
from quant.models.calibration import ProbabilityCalibration
from quant.services.market_tools import MarketTools


# ---------------------------------------------------------------------------
# kelly_fraction
# ---------------------------------------------------------------------------


def test_kelly_fraction_positive_ev():
    """kelly_fraction must return a positive value when EV > 0."""
    result = kelly_fraction(0.55, 2.1)
    assert result > 0, f"Expected positive Kelly fraction, got {result}"


def test_kelly_fraction_zero_for_negative_ev():
    """kelly_fraction must return 0.0 when EV is negative (p=0.30, odds=2.1)."""
    result = kelly_fraction(0.30, 2.1)
    assert result == 0.0, f"Expected 0.0 for negative EV, got {result}"


# ---------------------------------------------------------------------------
# expected_value
# ---------------------------------------------------------------------------


def test_expected_value_positive():
    """EV must be positive when the model probability beats the implied odds."""
    result = expected_value(0.55, 2.1)
    assert result > 0, f"Expected positive EV, got {result}"


def test_expected_value_negative():
    """EV must be negative when the model probability is below the break-even."""
    result = expected_value(0.30, 2.1)
    assert result < 0, f"Expected negative EV, got {result}"


# ---------------------------------------------------------------------------
# sharpe_ratio
# ---------------------------------------------------------------------------


def test_sharpe_ratio():
    """sharpe_ratio must return a positive value for a positive-EV bet."""
    result = sharpe_ratio(0.55, 0.155)
    assert result > 0, f"Expected positive Sharpe ratio, got {result}"


# ---------------------------------------------------------------------------
# MatchRanker.score_one
# ---------------------------------------------------------------------------


def test_match_ranker_score_one():
    """
    score_one() must return a non-None RankedBet with a valid tier and the
    correct match_id when given a clearly positive-EV record.
    """
    ranker = MatchRanker(bankroll=1000.0, kelly_scale=0.25)
    record = {
        "match_id": "1001",
        "market": "home",
        "probability": 0.55,
        "odds": 2.1,
        "decision": "BET",
        "confidence": 0.65,
        "agreement": 0.72,
        "model_edge": 0.05,
        "market_edge": 0.03,
    }
    result = ranker.score_one(record)

    assert result is not None, "score_one() returned None for a positive-EV record"
    assert result.tier in ("S", "A", "B", "C"), (
        f"Unexpected tier '{result.tier}'; expected one of S/A/B/C"
    )
    assert result.match_id == "1001", (
        f"Expected match_id '1001', got '{result.match_id}'"
    )


def test_match_ranker_excludes_no_value():
    """
    score_one() must return None for a bet with negative EV
    (probability=0.30, odds=2.1 → EV < 0 → tier 'X' → excluded).
    """
    ranker = MatchRanker(bankroll=1000.0)
    record = {
        "match_id": "1001",
        "market": "home",
        "probability": 0.30,
        "odds": 2.1,
        "decision": "BET",
    }
    result = ranker.score_one(record)
    assert result is None, (
        f"Expected None for a negative-EV bet, got {result}"
    )


# ---------------------------------------------------------------------------
# ProbabilityCalibration
# ---------------------------------------------------------------------------


def test_probability_calibration_sums_to_one():
    """
    calibrate_three_way() must return probabilities that sum to exactly 1.0
    (within floating-point tolerance).
    """
    cal = ProbabilityCalibration()
    raw = {"home_win": 0.5, "draw": 0.3, "away_win": 0.2}
    result = cal.calibrate_three_way(raw)

    total = result["home_win"] + result["draw"] + result["away_win"]
    assert total == pytest.approx(1.0, abs=1e-9), (
        f"Calibrated probabilities sum to {total}, expected 1.0"
    )


def test_probability_calibration_shrinks_toward_uniform():
    """
    Shrinkage toward the uniform prior (1/3) must reduce a home probability of
    0.5 to below 0.5 after calibration.
    """
    cal = ProbabilityCalibration(shrink=0.08)
    raw = {"home_win": 0.5, "draw": 0.3, "away_win": 0.2}
    result = cal.calibrate_three_way(raw)

    assert result["home_win"] < 0.5, (
        f"Expected calibrated home_win < 0.5 due to shrinkage, "
        f"got {result['home_win']}"
    )


# ---------------------------------------------------------------------------
# MarketTools
# ---------------------------------------------------------------------------


def test_market_tools_fair_odds():
    """
    fair_odds(0.5) must return approximately 2.0 (reciprocal of probability).
    """
    mt = MarketTools()
    result = mt.fair_odds(0.5)
    assert result == pytest.approx(2.0, rel=1e-6), (
        f"Expected fair_odds(0.5) ≈ 2.0, got {result}"
    )

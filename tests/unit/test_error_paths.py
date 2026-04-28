"""
Boundary value and error-path tests — 30+ cases.

Covers kelly_fraction, expected_value, ProbabilityCalibration,
MatchRanker, CsvExporter, MarketTools, AgreementEngine,
QuantConfidenceEngine, QuantNoBetFilter, EloEngine, FormEngine, PoissonEngine.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from export.csv_exporter import CsvExporter
from quant.models.calibration import ProbabilityCalibration
from quant.models.elo_engine import EloEngine
from quant.models.form_engine import FormEngine
from quant.models.poisson_engine import PoissonEngine
from quant.services.agreement_engine import AgreementEngine
from quant.services.confidence_engine import QuantConfidenceEngine
from quant.services.market_tools import MarketTools
from quant.services.no_bet_filter import QuantNoBetFilter
from ranking.match_ranker import (
    MatchRanker,
    expected_value,
    kelly_fraction,
    sharpe_ratio,
)

# ---------------------------------------------------------------------------
# kelly_fraction — boundary values
# ---------------------------------------------------------------------------


def test_kelly_fraction_extreme_probability_zero():
    assert kelly_fraction(0.0, 2.0) == 0.0


def test_kelly_fraction_extreme_probability_one():
    result = kelly_fraction(1.0, 2.0)
    assert isinstance(result, float)
    assert 0.0 <= result <= 1.0


def test_kelly_fraction_minimum_odds():
    assert kelly_fraction(0.5, 1.0) == pytest.approx(0.0)


def test_kelly_fraction_negative_ev_returns_zero():
    assert kelly_fraction(0.20, 2.0) == pytest.approx(0.0)


def test_kelly_fraction_capped_at_one():
    assert kelly_fraction(0.9999, 100.0) <= 1.0


def test_kelly_fraction_very_small_odds():
    result = kelly_fraction(0.99, 1.001)
    assert result >= 0.0
    assert not math.isnan(result)


def test_kelly_fraction_large_odds():
    result = kelly_fraction(0.5, 1000.0)
    assert 0.0 <= result <= 1.0
    assert not math.isnan(result)


# ---------------------------------------------------------------------------
# expected_value — boundary values
# ---------------------------------------------------------------------------


def test_expected_value_zero_probability():
    assert expected_value(0.0, 2.0) < 0


def test_expected_value_probability_one():
    assert expected_value(1.0, 2.0) > 0


def test_expected_value_zero_odds_yields_negative():
    assert expected_value(0.5, 0.0) < 0


def test_expected_value_fair_odds_near_zero():
    assert expected_value(0.5, 2.0) == pytest.approx(0.0, abs=0.01)


def test_expected_value_large_odds():
    result = expected_value(0.1, 15.0)
    assert not math.isnan(result)
    assert not math.isinf(result)


# ---------------------------------------------------------------------------
# sharpe_ratio — boundary values
# ---------------------------------------------------------------------------


def test_sharpe_ratio_zero_ev():
    assert sharpe_ratio(0.5, 0.0) == pytest.approx(0.0, abs=1e-9)


def test_sharpe_ratio_positive_ev():
    assert sharpe_ratio(0.6, 0.2) > 0


def test_sharpe_ratio_extreme_probability():
    result = sharpe_ratio(0.001, 0.5)
    assert not math.isnan(result)
    assert not math.isinf(result)


# ---------------------------------------------------------------------------
# ProbabilityCalibration — edge cases
# ---------------------------------------------------------------------------


def test_calibration_zero_inputs_returns_uniform():
    cal = ProbabilityCalibration()
    result = cal.calibrate_three_way({"home_win": 0, "draw": 0, "away_win": 0})
    assert result["home_win"] == pytest.approx(1 / 3, abs=1e-9)
    assert result["draw"] == pytest.approx(1 / 3, abs=1e-9)
    assert result["away_win"] == pytest.approx(1 / 3, abs=1e-9)


def test_calibration_binary_clamps_below_zero():
    assert ProbabilityCalibration().calibrate_binary(-0.5) > 0


def test_calibration_binary_clamps_above_one():
    assert ProbabilityCalibration().calibrate_binary(1.5) < 1


def test_calibration_three_way_output_sums_to_one():
    cal = ProbabilityCalibration()
    result = cal.calibrate_three_way({"home_win": 0.5, "draw": 0.25, "away_win": 0.25})
    assert sum(result.values()) == pytest.approx(1.0, abs=1e-6)


def test_calibration_three_way_all_ones_still_valid():
    cal = ProbabilityCalibration()
    result = cal.calibrate_three_way({"home_win": 1.0, "draw": 1.0, "away_win": 1.0})
    assert sum(result.values()) == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# MatchRanker.score_one — edge cases
# ---------------------------------------------------------------------------


def test_match_ranker_handles_missing_fields():
    assert MatchRanker().score_one({"probability": 0.55}) is None


def test_match_ranker_handles_nan_probability():
    assert MatchRanker().score_one({"probability": float("nan"), "odds": 2.1}) is None


def test_match_ranker_score_one_zero_probability():
    assert (
        MatchRanker().score_one({"probability": 0.0, "odds": 2.1, "decision": "BET"})
        is None
    )


def test_match_ranker_score_one_below_min_odds():
    assert (
        MatchRanker().score_one({"probability": 0.9, "odds": 1.02, "decision": "BET"})
        is None
    )


def test_match_ranker_rank_empty_list():
    assert MatchRanker().rank([]) == []


def test_match_ranker_rank_as_dicts_empty():
    assert MatchRanker().rank_as_dicts([]) == []


def test_match_ranker_all_no_bet_excluded_by_default():
    ranker = MatchRanker(include_no_bet=False)
    assert ranker.rank([{"probability": 0.6, "odds": 2.2, "decision": "NO_BET"}]) == []


def test_match_ranker_include_no_bet_flag():
    ranker = MatchRanker(include_no_bet=True)
    result = ranker.rank([{"probability": 0.6, "odds": 2.2, "decision": "NO_BET"}])
    assert len(result) == 1


# ---------------------------------------------------------------------------
# CsvExporter — edge cases
# ---------------------------------------------------------------------------


def test_csv_exporter_empty_records(tmp_path):
    exporter = CsvExporter(output_dir=str(tmp_path))
    result = exporter.export([], mode="full")
    assert result is not None
    assert Path(result).exists()


# ---------------------------------------------------------------------------
# MarketTools — boundary values
# ---------------------------------------------------------------------------


def test_market_tools_zero_odds_all_returns_uniform():
    mt = MarketTools()
    result = mt.normalize_implied_probs_1x2({"home": 0, "draw": 0, "away": 0})
    assert result["home_win"] == pytest.approx(1 / 3, abs=1e-9)


def test_market_tools_fair_odds_zero_prob():
    assert MarketTools().fair_odds(0.0) == pytest.approx(999.0)


def test_market_tools_fair_odds_negative_prob():
    assert MarketTools().fair_odds(-0.5) == pytest.approx(999.0)


def test_market_tools_edge_zero_bookmaker_odds():
    assert MarketTools().edge(0.55, 0.0) == pytest.approx(0.0)


def test_market_tools_edge_none_bookmaker_odds():
    assert MarketTools().edge(0.55, None) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# AgreementEngine — boundary values
# ---------------------------------------------------------------------------


def test_agreement_engine_empty_list():
    assert AgreementEngine().three_way_agreement([]) == 0.0


def test_agreement_engine_single_model_nonnegative():
    p = {"home_win": 0.5, "draw": 0.25, "away_win": 0.25}
    assert AgreementEngine().three_way_agreement([p]) >= 0.0


def test_agreement_engine_missing_keys_no_crash():
    score = AgreementEngine().three_way_agreement([{}, {}])
    assert isinstance(score, float)


# ---------------------------------------------------------------------------
# QuantConfidenceEngine — boundary values
# ---------------------------------------------------------------------------


def test_confidence_all_zeros():
    assert QuantConfidenceEngine().score(0.0, 0.0, 0.0, 0.0) == pytest.approx(0.0)


def test_confidence_negative_inputs_clamped():
    assert QuantConfidenceEngine().score(-1.0, -1.0, -1.0, -1.0) >= 0.0


def test_confidence_output_bounded():
    result = QuantConfidenceEngine().score(0.7, 0.10, 0.8, 0.9)
    assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# QuantNoBetFilter — boundary values
# ---------------------------------------------------------------------------


def test_no_bet_filter_all_zeros_is_no_bet():
    assert QuantNoBetFilter().decide(0.0, 0.0, 0.0, 0.0) == "NO_BET"


def test_no_bet_filter_perfect_inputs_is_bet():
    assert QuantNoBetFilter().decide(1.0, 1.0, 1.0, 1.0) == "BET"


# ---------------------------------------------------------------------------
# EloEngine — boundary values
# ---------------------------------------------------------------------------


def test_elo_unknown_team_returns_base_rating():
    assert EloEngine(base_rating=1500.0).get_rating("Unknown Team") == 1500.0


def test_elo_fit_empty_list_no_crash():
    elo = EloEngine()
    elo.fit([])
    assert elo.ratings == {}


def test_elo_expected_score_equal_ratings_is_half():
    elo = EloEngine()
    assert elo.expected_score(1500.0, 1500.0) == pytest.approx(0.5, abs=0.01)


# ---------------------------------------------------------------------------
# FormEngine — boundary values
# ---------------------------------------------------------------------------


def test_form_unknown_team_returns_half():
    form = FormEngine()
    form.fit([])
    assert form.get_form("Ghost FC") == 0.5


def test_form_fit_empty_list_no_crash():
    form = FormEngine()
    form.fit([])
    assert form.team_form == {}


# ---------------------------------------------------------------------------
# PoissonEngine — boundary values
# ---------------------------------------------------------------------------


def test_poisson_empty_fit_baseline_lambdas():
    engine = PoissonEngine()
    engine.fit([])
    lh, la = engine.expected_goals("X", "Y")
    assert lh == pytest.approx(1.35, abs=0.01)
    assert la == pytest.approx(1.10, abs=0.01)


def test_poisson_probabilities_sum_to_one_after_empty_fit():
    engine = PoissonEngine()
    engine.fit([])
    p = engine.probabilities_1x2("X", "Y")
    assert sum(p.values()) == pytest.approx(1.0, abs=1e-6)


def test_poisson_lambda_floor_prevents_zero():
    engine = PoissonEngine()
    # Even a very weak team should not produce lambda=0
    engine.fit(
        [
            {
                "home_team": "Weak",
                "away_team": "Strong",
                "home_goals": 0,
                "away_goals": 10,
            }
        ]
        * 20
    )
    lh, _ = engine.expected_goals("Weak", "Strong")
    assert lh >= 0.2

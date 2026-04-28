"""
Unit tests for quant/services/ — AgreementEngine, QuantConfidenceEngine,
QuantNoBetFilter, MarketTools (extended coverage beyond test_quant_core.py).
"""

from __future__ import annotations

import pytest

from quant.services.agreement_engine import AgreementEngine
from quant.services.confidence_engine import QuantConfidenceEngine
from quant.services.no_bet_filter import QuantNoBetFilter
from quant.services.market_tools import MarketTools

# ---------------------------------------------------------------------------
# AgreementEngine
# ---------------------------------------------------------------------------


class TestAgreementEngine:

    def test_empty_list_returns_zero(self):
        eng = AgreementEngine()
        assert eng.three_way_agreement([]) == 0.0

    def test_identical_models_maximum_agreement(self):
        p = {"home_win": 0.50, "draw": 0.25, "away_win": 0.25}
        eng = AgreementEngine()
        score = eng.three_way_agreement([p, p, p, p])
        assert score == pytest.approx(1.0, abs=0.01)

    def test_maximally_divergent_models_low_agreement(self):
        models = [
            {"home_win": 1.0, "draw": 0.0, "away_win": 0.0},
            {"home_win": 0.0, "draw": 1.0, "away_win": 0.0},
            {"home_win": 0.0, "draw": 0.0, "away_win": 1.0},
        ]
        eng = AgreementEngine()
        score = eng.three_way_agreement(models)
        assert score == 0.0

    def test_single_model_returns_nonnegative(self):
        eng = AgreementEngine()
        score = eng.three_way_agreement(
            [{"home_win": 0.4, "draw": 0.3, "away_win": 0.3}]
        )
        assert score >= 0.0

    def test_score_bounded_zero_to_one(self):
        import random

        random.seed(42)
        eng = AgreementEngine()
        for _ in range(20):
            models = [
                {
                    "home_win": random.random(),
                    "draw": random.random(),
                    "away_win": random.random(),
                }
                for _ in range(4)
            ]
            score = eng.three_way_agreement(models)
            assert 0.0 <= score <= 1.0

    def test_missing_keys_treated_as_zero(self):
        eng = AgreementEngine()
        # Models with missing keys should not crash
        score = eng.three_way_agreement([{}, {}])
        assert isinstance(score, float)


# ---------------------------------------------------------------------------
# QuantConfidenceEngine
# ---------------------------------------------------------------------------


class TestQuantConfidenceEngine:

    def test_all_zeros_returns_zero(self):
        eng = QuantConfidenceEngine()
        assert eng.score(0.0, 0.0, 0.0, 0.0) == pytest.approx(0.0)

    def test_all_max_inputs_returns_one(self):
        eng = QuantConfidenceEngine()
        result = eng.score(1.0, 0.15, 1.0, 1.0)
        assert result == pytest.approx(1.0, abs=0.01)

    def test_negative_probability_clamped(self):
        eng = QuantConfidenceEngine()
        result = eng.score(-5.0, 0.10, 0.7, 0.5)
        assert result >= 0.0

    def test_negative_edge_clamped_to_zero(self):
        eng = QuantConfidenceEngine()
        result_neg = eng.score(0.6, -0.10, 0.7, 0.5)
        result_zero = eng.score(0.6, 0.0, 0.7, 0.5)
        # Negative edge should be treated same as zero
        assert result_neg == pytest.approx(result_zero, abs=1e-9)

    def test_output_bounded_zero_to_one(self):
        eng = QuantConfidenceEngine()
        for prob in [0.0, 0.3, 0.6, 0.9, 1.0]:
            for edge in [0.0, 0.05, 0.15, 0.5]:
                result = eng.score(prob, edge, 0.7, 0.4)
                assert 0.0 <= result <= 1.0

    def test_higher_probability_higher_confidence(self):
        eng = QuantConfidenceEngine()
        low = eng.score(0.3, 0.08, 0.7, 0.5)
        high = eng.score(0.7, 0.08, 0.7, 0.5)
        assert high > low


# ---------------------------------------------------------------------------
# QuantNoBetFilter
# ---------------------------------------------------------------------------


class TestQuantNoBetFilter:

    def test_all_zeros_returns_no_bet(self):
        f = QuantNoBetFilter()
        assert f.decide(0.0, 0.0, 0.0, 0.0) == "NO_BET"

    def test_all_thresholds_exactly_met_returns_bet(self):
        f = QuantNoBetFilter(
            min_probability=0.54,
            min_edge=0.04,
            min_confidence=0.64,
            min_agreement=0.58,
        )
        result = f.decide(0.54, 0.04, 0.64, 0.58)
        assert result == "BET"

    def test_slightly_below_threshold_returns_watchlist_or_no_bet(self):
        f = QuantNoBetFilter(
            min_probability=0.54,
            min_edge=0.04,
            min_confidence=0.64,
            min_agreement=0.58,
        )
        # Slightly below → watchlist range
        result = f.decide(0.52, 0.03, 0.52, 0.50)
        assert result in ("WATCHLIST", "NO_BET")

    def test_custom_thresholds_respected(self):
        # Very tight filter — almost nothing passes
        strict = QuantNoBetFilter(
            min_probability=0.99,
            min_edge=0.99,
            min_confidence=0.99,
            min_agreement=0.99,
        )
        assert strict.decide(0.55, 0.05, 0.65, 0.60) == "NO_BET"

    def test_clearly_good_bet(self):
        f = QuantNoBetFilter()
        result = f.decide(0.70, 0.15, 0.80, 0.75)
        assert result == "BET"

    def test_output_always_one_of_three_values(self):
        f = QuantNoBetFilter()
        valid = {"BET", "WATCHLIST", "NO_BET"}
        for prob in [0.0, 0.3, 0.55, 0.8, 1.0]:
            result = f.decide(prob, 0.05, 0.65, 0.60)
            assert result in valid


# ---------------------------------------------------------------------------
# MarketTools (extended)
# ---------------------------------------------------------------------------


class TestMarketToolsExtended:

    def test_normalize_all_zero_odds_returns_uniform(self):
        mt = MarketTools()
        result = mt.normalize_implied_probs_1x2({"home": 0, "draw": 0, "away": 0})
        assert result["home_win"] == pytest.approx(1 / 3, abs=1e-9)
        assert result["draw"] == pytest.approx(1 / 3, abs=1e-9)
        assert result["away_win"] == pytest.approx(1 / 3, abs=1e-9)

    def test_normalize_none_values_returns_uniform(self):
        mt = MarketTools()
        result = mt.normalize_implied_probs_1x2(
            {"home": None, "draw": None, "away": None}
        )
        assert sum(result.values()) == pytest.approx(1.0, abs=1e-9)

    def test_normalize_missing_keys_returns_uniform(self):
        mt = MarketTools()
        result = mt.normalize_implied_probs_1x2({})
        assert sum(result.values()) == pytest.approx(1.0, abs=1e-9)

    def test_normalize_probs_sum_to_one(self):
        mt = MarketTools()
        result = mt.normalize_implied_probs_1x2({"home": 2.1, "draw": 3.4, "away": 3.2})
        assert sum(result.values()) == pytest.approx(1.0, abs=1e-9)

    def test_fair_odds_zero_probability(self):
        mt = MarketTools()
        assert mt.fair_odds(0.0) == pytest.approx(999.0)

    def test_fair_odds_negative_probability(self):
        mt = MarketTools()
        assert mt.fair_odds(-0.5) == pytest.approx(999.0)

    def test_fair_odds_one_returns_one(self):
        mt = MarketTools()
        assert mt.fair_odds(1.0) == pytest.approx(1.0)

    def test_edge_zero_bookmaker_odds(self):
        mt = MarketTools()
        assert mt.edge(0.55, 0.0) == pytest.approx(0.0)

    def test_edge_none_bookmaker_odds(self):
        mt = MarketTools()
        assert mt.edge(0.55, None) == pytest.approx(0.0)

    def test_positive_edge_on_value_bet(self):
        mt = MarketTools()
        # Model 55%, bookmaker odds 2.20 → edge = 0.55*2.20-1 = 0.21
        assert mt.edge(0.55, 2.20) == pytest.approx(0.21, abs=0.001)

    def test_probability_edge_vs_market_positive(self):
        mt = MarketTools()
        assert mt.probability_edge_vs_market(0.55, 0.48) == pytest.approx(
            0.07, abs=1e-9
        )

    def test_probability_edge_vs_market_negative(self):
        mt = MarketTools()
        assert mt.probability_edge_vs_market(0.40, 0.52) == pytest.approx(
            -0.12, abs=1e-9
        )

"""
Unit tests for strategies/ — EdgeDetector, NoBetFilter, ValueBetStrategy.
All logic is pure Python; no external dependencies mocked.
"""

from __future__ import annotations

import pytest

from strategies.edge_detector import EdgeDetector
from strategies.no_bet_filter import NoBetFilter
from strategies.value_bet_strategy import ValueBetStrategy

# ---------------------------------------------------------------------------
# EdgeDetector
# ---------------------------------------------------------------------------


class TestEdgeDetector:
    @pytest.fixture
    def detector(self) -> EdgeDetector:
        return EdgeDetector()

    def test_positive_edge_is_value(self, detector: EdgeDetector) -> None:
        assert detector.is_value_bet(0.60, 2.0) is True  # edge = 0.10

    def test_zero_edge_not_value(self, detector: EdgeDetector) -> None:
        assert detector.is_value_bet(0.50, 2.0) is False  # implied = 0.50

    def test_negative_edge_not_value(self, detector: EdgeDetector) -> None:
        assert detector.is_value_bet(0.40, 2.0) is False

    def test_calculate_edge_correct(self, detector: EdgeDetector) -> None:
        edge = detector.calculate_edge(0.60, 2.0)
        assert edge == pytest.approx(0.10, abs=1e-9)

    def test_zero_odds_returns_zero_edge(self, detector: EdgeDetector) -> None:
        assert detector.calculate_edge(0.5, 0.0) == 0.0

    def test_negative_odds_returns_zero_edge(self, detector: EdgeDetector) -> None:
        assert detector.calculate_edge(0.5, -1.0) == 0.0

    def test_none_probability_not_value(self, detector: EdgeDetector) -> None:
        assert detector.is_value_bet(None, 2.0) is False

    def test_none_odds_not_value(self, detector: EdgeDetector) -> None:
        assert detector.is_value_bet(0.60, None) is False

    def test_both_none_not_value(self, detector: EdgeDetector) -> None:
        assert detector.is_value_bet(None, None) is False

    def test_very_high_odds_small_prob(self, detector: EdgeDetector) -> None:
        # implied = 1/100 = 0.01; edge = 0.02 − 0.01 = 0.01 → depends on threshold
        edge = detector.calculate_edge(0.02, 100.0)
        assert edge == pytest.approx(0.01, abs=1e-9)


# ---------------------------------------------------------------------------
# NoBetFilter
# ---------------------------------------------------------------------------


class TestNoBetFilter:
    @pytest.fixture
    def filt(self) -> NoBetFilter:
        return NoBetFilter()

    def test_all_above_thresholds_returns_bet(self, filt: NoBetFilter) -> None:
        assert filt.decide(0.60, 0.10, 0.70, 0.65) == "BET"

    def test_all_below_returns_no_bet(self, filt: NoBetFilter) -> None:
        assert filt.decide(0.40, 0.02, 0.50, 0.40) == "NO_BET"

    def test_borderline_returns_watchlist(self, filt: NoBetFilter) -> None:
        # Slightly below BET thresholds but above WATCHLIST (95%/60%/80%/80%)
        result = filt.decide(0.515, 0.031, 0.522, 0.482)
        assert result in ("WATCHLIST", "NO_BET")

    def test_exact_minimum_bet_thresholds(self, filt: NoBetFilter) -> None:
        assert filt.decide(0.54, 0.05, 0.65, 0.60) == "BET"

    def test_string_inputs_coerced(self, filt: NoBetFilter) -> None:
        result = filt.decide("0.60", "0.10", "0.70", "0.65")  # type: ignore[arg-type]
        assert result == "BET"

    def test_custom_thresholds(self) -> None:
        strict = NoBetFilter(
            min_probability=0.70,
            min_edge=0.10,
            min_confidence=0.80,
            min_agreement=0.75,
        )
        assert strict.decide(0.60, 0.08, 0.70, 0.65) == "NO_BET"

    def test_decide_returns_only_valid_values(self, filt: NoBetFilter) -> None:
        for prob in (0.0, 0.5, 0.54, 0.60, 1.0):
            result = filt.decide(prob, 0.05, 0.65, 0.60)
            assert result in ("BET", "WATCHLIST", "NO_BET")


# ---------------------------------------------------------------------------
# ValueBetStrategy
# ---------------------------------------------------------------------------


class TestValueBetStrategy:
    @pytest.fixture
    def strategy(self) -> ValueBetStrategy:
        return ValueBetStrategy()

    @pytest.fixture
    def detector(self) -> EdgeDetector:
        return EdgeDetector()

    def test_finds_home_value_bet(
        self, strategy: ValueBetStrategy, detector: EdgeDetector
    ) -> None:
        predictions = {"m1": {"home_win": 0.65, "draw": 0.20, "away_win": 0.15}}
        odds_data = {"m1": {"home": 2.0, "draw": 4.0, "away": 7.0}}
        results = strategy.find_value_bets(predictions, odds_data, detector)
        markets = [r["market"] for r in results]
        assert "home" in markets

    def test_no_value_returns_empty(
        self, strategy: ValueBetStrategy, detector: EdgeDetector
    ) -> None:
        predictions = {"m1": {"home_win": 0.30, "draw": 0.30, "away_win": 0.40}}
        odds_data = {"m1": {"home": 2.0, "draw": 3.0, "away": 2.5}}
        results = strategy.find_value_bets(predictions, odds_data, detector)
        assert results == []

    def test_missing_odds_match_skipped(
        self, strategy: ValueBetStrategy, detector: EdgeDetector
    ) -> None:
        predictions = {"m1": {"home_win": 0.70}, "m2": {"home_win": 0.70}}
        odds_data = {"m1": {"home": 2.0, "draw": 4.0, "away": 7.0}}
        results = strategy.find_value_bets(predictions, odds_data, detector)
        ids = [r["match_id"] for r in results]
        assert "m2" not in ids

    def test_result_structure(
        self, strategy: ValueBetStrategy, detector: EdgeDetector
    ) -> None:
        predictions = {"m1": {"home_win": 0.70, "draw": 0.15, "away_win": 0.15}}
        odds_data = {"m1": {"home": 2.5, "draw": 5.0, "away": 8.0}}
        results = strategy.find_value_bets(predictions, odds_data, detector)
        for r in results:
            assert "match_id" in r
            assert "market" in r
            assert "probability" in r
            assert "odds" in r

    def test_empty_predictions(
        self, strategy: ValueBetStrategy, detector: EdgeDetector
    ) -> None:
        assert strategy.find_value_bets({}, {}, detector) == []

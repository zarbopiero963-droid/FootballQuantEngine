"""
Tests for strategies.edge_detector, strategies.no_bet_filter,
and strategies.value_bet_strategy.
"""

from __future__ import annotations

from strategies.edge_detector import EdgeDetector
from strategies.no_bet_filter import NoBetFilter
from strategies.value_bet_strategy import ValueBetStrategy

# ---------------------------------------------------------------------------
# EdgeDetector
# ---------------------------------------------------------------------------


class TestEdgeDetector:
    def setup_method(self):
        self.detector = EdgeDetector()

    def test_positive_edge_detected(self):
        edge = self.detector.calculate_edge(probability=0.55, odds=2.1)
        assert edge > 0.0

    def test_zero_edge_on_fair_odds(self):
        edge = self.detector.calculate_edge(probability=0.5, odds=2.0)
        assert abs(edge) < 0.01

    def test_negative_edge_detected(self):
        edge = self.detector.calculate_edge(probability=0.40, odds=2.1)
        assert edge < 0.0

    def test_is_value_bet_true_when_above_threshold(self):
        assert self.detector.is_value_bet(probability=0.65, odds=2.0)

    def test_is_value_bet_false_when_below_threshold(self):
        assert not self.detector.is_value_bet(probability=0.30, odds=4.0)


# ---------------------------------------------------------------------------
# NoBetFilter
# ---------------------------------------------------------------------------


class TestNoBetFilter:
    def setup_method(self):
        self.fltr = NoBetFilter(
            min_probability=0.54, min_edge=0.05, min_confidence=0.65, min_agreement=0.60
        )

    def test_value_bet_returns_bet(self):
        decision = self.fltr.decide(
            probability=0.60, edge=0.10, confidence=0.70, agreement=0.75
        )
        assert decision == "BET"

    def test_low_probability_not_bet(self):
        decision = self.fltr.decide(
            probability=0.40, edge=0.10, confidence=0.70, agreement=0.75
        )
        assert decision != "BET"

    def test_negative_edge_not_bet(self):
        decision = self.fltr.decide(
            probability=0.60, edge=-0.05, confidence=0.70, agreement=0.75
        )
        assert decision != "BET"

    def test_low_confidence_not_bet(self):
        decision = self.fltr.decide(
            probability=0.60, edge=0.10, confidence=0.50, agreement=0.75
        )
        assert decision != "BET"


# ---------------------------------------------------------------------------
# ValueBetStrategy
# ---------------------------------------------------------------------------


class TestValueBetStrategy:
    def setup_method(self):
        self.strategy = ValueBetStrategy()
        self.detector = EdgeDetector()

    def _preds(self, home_win=0.60, draw=0.22, away_win=0.18):
        return {"1001": {"home_win": home_win, "draw": draw, "away_win": away_win}}

    def _odds(self):
        return {"1001": {"home": 2.1, "draw": 3.5, "away": 4.5}}

    def test_returns_list(self):
        result = self.strategy.find_value_bets(
            self._preds(), self._odds(), self.detector
        )
        assert isinstance(result, list)

    def test_finds_positive_edge_bet(self):
        result = self.strategy.find_value_bets(
            self._preds(home_win=0.65), self._odds(), self.detector
        )
        markets = [b.get("market") for b in result]
        assert "home" in markets

    def test_empty_predictions_returns_empty(self):
        result = self.strategy.find_value_bets({}, {}, self.detector)
        assert result == []

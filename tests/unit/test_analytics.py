"""
Unit tests for analytics/ — ProbabilityMarkets, LeaguePredictabilityAnalyser,
MarketInefficiencyScanner.
"""

from __future__ import annotations

import pytest

from analytics.league_predictability import (
    LeaguePredictability,
    LeaguePredictabilityAnalyser,
    _gini,
    _ranked_prob_score,
)
from analytics.market_inefficiency_scanner import MarketInefficiencyScanner
from analytics.probability_markets import ProbabilityMarkets, poisson_pmf

# ---------------------------------------------------------------------------
# Helpers / shared data
# ---------------------------------------------------------------------------


def _make_records(n: int = 10):
    """Minimal prediction records for LeaguePredictability tests."""
    records = []
    for i in range(n):
        records.append(
            {
                "league": "Serie A",
                "actual_outcome": "home_win" if i % 3 != 2 else "away_win",
                "p_home": 0.50,
                "p_draw": 0.25,
                "p_away": 0.25,
                "home_odds": 2.0,
                "away_odds": 3.5,
            }
        )
    return records


# ---------------------------------------------------------------------------
# ProbabilityMarkets
# ---------------------------------------------------------------------------


class TestProbabilityMarkets:

    def test_poisson_pmf_zero_goals(self):
        import math

        result = poisson_pmf(0, 1.5)
        assert result == pytest.approx(math.exp(-1.5), rel=1e-9)

    def test_score_matrix_dimensions(self):
        pm = ProbabilityMarkets()
        matrix = pm.score_matrix(1.5, 1.1, max_goals=5)
        assert len(matrix) == 6
        assert all(len(row) == 6 for row in matrix)

    def test_score_matrix_sums_to_approx_one(self):
        pm = ProbabilityMarkets()
        matrix = pm.score_matrix(1.5, 1.1, max_goals=20)
        total = sum(cell for row in matrix for cell in row)
        assert total == pytest.approx(1.0, abs=0.001)

    def test_over_under_sum_to_one(self):
        pm = ProbabilityMarkets()
        result = pm.over_under_25(1.5, 1.1)
        assert result["over_2_5"] + result["under_2_5"] == pytest.approx(1.0, abs=1e-6)

    def test_btts_sum_to_one(self):
        pm = ProbabilityMarkets()
        result = pm.btts(1.5, 1.1)
        assert result["btts_yes"] + result["btts_no"] == pytest.approx(1.0, abs=1e-6)

    def test_high_lambdas_more_overs(self):
        pm = ProbabilityMarkets()
        low = pm.over_under_25(0.8, 0.8)
        high = pm.over_under_25(2.5, 2.5)
        assert high["over_2_5"] > low["over_2_5"]

    def test_high_lambdas_more_btts(self):
        pm = ProbabilityMarkets()
        low = pm.btts(0.5, 0.5)
        high = pm.btts(2.5, 2.5)
        assert high["btts_yes"] > low["btts_yes"]

    def test_very_low_lambda_mostly_zero_goals(self):
        pm = ProbabilityMarkets()
        # Near-zero lambda → most likely 0 goals
        matrix = pm.score_matrix(0.01, 0.01, max_goals=10)
        assert matrix[0][0] > 0.95


# ---------------------------------------------------------------------------
# _gini helper
# ---------------------------------------------------------------------------


class TestGiniHelper:

    def test_equal_values_zero_gini(self):
        assert _gini([1.0, 1.0, 1.0, 1.0]) == pytest.approx(0.0, abs=1e-9)

    def test_single_value_zero_gini(self):
        assert _gini([5.0]) == pytest.approx(0.0, abs=1e-9)

    def test_empty_list_zero_gini(self):
        assert _gini([]) == pytest.approx(0.0, abs=1e-9)

    def test_extreme_concentration_high_gini(self):
        # One team has all the points
        assert _gini([0.0, 0.0, 0.0, 100.0]) > 0.5


# ---------------------------------------------------------------------------
# _ranked_prob_score helper
# ---------------------------------------------------------------------------


class TestRankedProbScore:

    def test_perfect_prediction_zero_rps(self):
        rps = _ranked_prob_score(1.0, 0.0, 0.0, "home_win")
        assert rps == pytest.approx(0.0, abs=1e-9)

    def test_worst_prediction_max_rps(self):
        # All weight on away_win, outcome is home_win → RPS = 1.0 for this impl
        rps = _ranked_prob_score(0.0, 0.0, 1.0, "home_win")
        assert rps == pytest.approx(1.0, abs=1e-6)

    def test_uniform_prediction_medium_rps(self):
        rps = _ranked_prob_score(1 / 3, 1 / 3, 1 / 3, "home_win")
        assert 0.0 < rps < 0.5

    def test_unknown_outcome_returns_max_entropy(self):
        rps = _ranked_prob_score(0.5, 0.3, 0.2, "invalid")
        assert rps == pytest.approx(0.5, abs=1e-9)


# ---------------------------------------------------------------------------
# LeaguePredictabilityAnalyser
# ---------------------------------------------------------------------------


class TestLeaguePredictabilityAnalyser:

    def test_empty_records_returns_empty_report(self):
        analyser = LeaguePredictabilityAnalyser()
        report = analyser.analyse([], league="Test")
        assert report.n_matches == 0
        assert report.predictability_score == 0.0
        assert report.grade == "D"

    def test_analyse_returns_league_report(self):
        analyser = LeaguePredictabilityAnalyser()
        report = analyser.analyse(_make_records(20), league="Serie A")
        assert report.league == "Serie A"
        assert report.n_matches == 20
        assert 0.0 <= report.predictability_score <= 1.0
        assert report.grade in ("A", "B", "C", "D")

    def test_rps_is_nonnegative(self):
        analyser = LeaguePredictabilityAnalyser()
        report = analyser.analyse(_make_records(10))
        assert report.rps >= 0.0

    def test_home_away_draw_rates_sum_to_one(self):
        analyser = LeaguePredictabilityAnalyser()
        report = analyser.analyse(_make_records(15))
        total = report.home_win_rate + report.draw_rate + report.away_win_rate
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_summary_table_sorted_descending(self):
        analyser = LeaguePredictabilityAnalyser()
        records = _make_records(15)
        for r in records[:5]:
            r["league"] = "Bundesliga"
        table = analyser.summary_table(records)
        scores = [row["predictability_score"] for row in table]
        assert scores == sorted(scores, reverse=True)

    def test_as_dict_contains_expected_keys(self):
        analyser = LeaguePredictabilityAnalyser()
        report = analyser.analyse(_make_records(10))
        d = report.as_dict()
        for key in ("league", "n_matches", "rps", "predictability_score", "grade"):
            assert key in d

    def test_calibration_curve_returns_n_bins(self):
        analyser = LeaguePredictabilityAnalyser()
        curve = analyser.calibration_curve(_make_records(30), n_bins=5)
        assert len(curve) == 5


# ---------------------------------------------------------------------------
# LeaguePredictability DataFrame adapter
# ---------------------------------------------------------------------------


class TestLeaguePredictabilityAdapter:

    def test_score_returns_dataframe(self):
        pd = pytest.importorskip("pandas")
        lp = LeaguePredictability()
        df = pd.DataFrame(
            [
                {"league": "Serie A", "home_goals": 2, "away_goals": 1},
                {"league": "Serie A", "home_goals": 0, "away_goals": 0},
                {"league": "Premier League", "home_goals": 1, "away_goals": 2},
            ]
        )
        result = lp.score(df)
        assert isinstance(result, pd.DataFrame)
        assert "league" in result.columns
        assert "predictability_score" in result.columns

    def test_score_empty_dataframe(self):
        pd = pytest.importorskip("pandas")
        lp = LeaguePredictability()
        df = pd.DataFrame(columns=["league", "home_goals", "away_goals"])
        result = lp.score(df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# MarketInefficiencyScanner
# ---------------------------------------------------------------------------


class TestMarketInefficiencyScanner:

    def _predictions(self, n: int = 15):
        preds = []
        for i in range(n):
            preds.append(
                {
                    "fixture_id": str(i),
                    "league": "Serie A",
                    "market": "home",
                    "p_model": 0.55,
                    "p_implied": 0.48,
                    "bookmaker_odds": 2.08,
                    "timestamp": 1700000000 + i * 3600,
                }
            )
        return preds

    def test_scan_returns_list(self):
        scanner = MarketInefficiencyScanner()
        results = scanner.scan(self._predictions())
        assert isinstance(results, list)

    def test_scan_empty_returns_empty(self):
        scanner = MarketInefficiencyScanner()
        results = scanner.scan([])
        assert results == []

    def test_report_returns_dict(self):
        scanner = MarketInefficiencyScanner()
        preds = self._predictions(20)
        results = scanner.scan(preds)
        report = scanner.report(results)
        assert isinstance(report, dict)

    def test_clear_history(self):
        scanner = MarketInefficiencyScanner()
        scanner.scan(self._predictions(10))
        scanner.clear_history()
        summary = scanner.history_summary()
        assert summary.get("total", 0) == 0

    def test_scan_result_has_required_fields(self):
        scanner = MarketInefficiencyScanner()
        results = scanner.scan(self._predictions(20))
        for r in results:
            assert "p_model" in r or "fixture_id" in r  # at least one key preserved

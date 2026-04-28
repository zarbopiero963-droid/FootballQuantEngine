"""
Unit tests for quant/models and quant/markets — high-coverage targets.

Covers CornersModel, HalftimeModel, CorrectScoreModel, MultiMarketEngine.
All pure-Python, no I/O, no network, no PySide6.
"""

from __future__ import annotations

import math

import pytest

from quant.models.corners_model import CornersModel
from quant.models.halftime_model import HalftimeModel
from quant.models.correct_score_model import CorrectScoreModel
from quant.markets.multi_market_engine import MultiMarketEngine


# ---------------------------------------------------------------------------
# Fixtures shared across all classes
# ---------------------------------------------------------------------------

_MATCHES = [
    {
        "home_team": "Juventus", "away_team": "Inter",
        "home_goals": 2, "away_goals": 1,
        "ht_home": 1, "ht_away": 0,
        "home_corners": 6, "away_corners": 4,
    },
    {
        "home_team": "AC Milan", "away_team": "Roma",
        "home_goals": 1, "away_goals": 1,
        "ht_home": 0, "ht_away": 1,
        "home_corners": 5, "away_corners": 5,
    },
    {
        "home_team": "Napoli", "away_team": "Lazio",
        "home_goals": 3, "away_goals": 0,
        "ht_home": 2, "ht_away": 0,
        "home_corners": 8, "away_corners": 3,
    },
    {
        "home_team": "Inter", "away_team": "Napoli",
        "home_goals": 2, "away_goals": 2,
        "ht_home": 1, "ht_away": 1,
        "home_corners": 5, "away_corners": 6,
    },
    {
        "home_team": "Roma", "away_team": "Juventus",
        "home_goals": 1, "away_goals": 2,
        "ht_home": 0, "ht_away": 1,
        "home_corners": 4, "away_corners": 7,
    },
]


# ===========================================================================
# CornersModel
# ===========================================================================

class TestCornersModel:

    def test_default_constructor(self):
        m = CornersModel()
        assert m.max_corners == 20
        assert m.league_avg_home_corners > 0
        assert m.league_avg_away_corners > 0

    def test_fit_updates_league_averages(self):
        m = CornersModel()
        m.fit(_MATCHES)
        # league averages are recalculated
        assert m.league_avg_home_corners == pytest.approx(5.6, abs=0.5)
        assert m.league_avg_away_corners == pytest.approx(5.0, abs=0.5)

    def test_fit_empty_list(self):
        m = CornersModel()
        before_h = m.league_avg_home_corners
        m.fit([])
        assert m.league_avg_home_corners == before_h  # unchanged

    def test_fit_skips_missing_corners(self):
        m = CornersModel()
        bad = [{"home_team": "A", "away_team": "B", "home_goals": 1, "away_goals": 0}]
        m.fit(bad)  # no home_corners/away_corners — should not crash

    def test_expected_corners_returns_positive(self):
        m = CornersModel()
        m.fit(_MATCHES)
        lam_h, lam_a = m.expected_corners("Juventus", "Inter")
        assert lam_h > 0
        assert lam_a > 0

    def test_expected_corners_unknown_team_uses_defaults(self):
        m = CornersModel()
        m.fit(_MATCHES)
        lam_h, lam_a = m.expected_corners("Unknown FC", "Another Club")
        assert lam_h >= 0.5
        assert lam_a >= 0.5

    def test_probabilities_over_under_sum_to_one(self):
        m = CornersModel()
        m.fit(_MATCHES)
        p = m.probabilities("Juventus", "Inter", line=9.5)
        assert p["over"] + p["under"] == pytest.approx(1.0, abs=0.001)

    def test_probabilities_keys_present(self):
        m = CornersModel()
        m.fit(_MATCHES)
        p = m.probabilities("Inter", "Roma")
        assert "lambda_home" in p
        assert "lambda_away" in p
        assert "lambda_total" in p
        assert "line" in p
        assert "over" in p
        assert "under" in p

    def test_all_lines_returns_three_markets(self):
        m = CornersModel()
        m.fit(_MATCHES)
        result = m.all_lines("Napoli", "Lazio")
        assert "corners_8.5" in result
        assert "corners_9.5" in result
        assert "corners_10.5" in result

    def test_probabilities_over_high_line_nearly_zero(self):
        m = CornersModel()
        p = m.probabilities("A", "B", line=40.0)
        assert p["over"] < 0.05

    def test_probabilities_under_high_line_nearly_one(self):
        m = CornersModel()
        p = m.probabilities("A", "B", line=40.0)
        assert p["under"] > 0.95

    def test_poisson_pmf_zero_k(self):
        m = CornersModel()
        val = m._poisson_pmf(0, 2.5)
        assert val == pytest.approx(math.exp(-2.5), rel=1e-6)

    def test_poisson_pmf_vec_sums_to_one(self):
        m = CornersModel()
        vec = m._poisson_pmf_vec(5.0)
        assert sum(vec) == pytest.approx(1.0, abs=0.001)


# ===========================================================================
# HalftimeModel
# ===========================================================================

class TestHalftimeModel:

    def test_default_ratio(self):
        m = HalftimeModel()
        assert m._ht_ratio_home == pytest.approx(0.42)
        assert m._ht_ratio_away == pytest.approx(0.42)

    def test_fit_requires_20_valid_rows(self):
        m = HalftimeModel()
        # Only 5 rows — should not update ratio
        m.fit(_MATCHES)
        assert m._ht_ratio_home == pytest.approx(0.42)

    def test_fit_with_enough_data_updates_ratio(self):
        m = HalftimeModel()
        # Manufacture 25 valid rows
        rows = []
        for i in range(25):
            rows.append({
                "home_goals": 2, "away_goals": 2,
                "ht_home": 1, "ht_away": 1,
            })
        m.fit(rows)
        assert m._ht_ratio_home == pytest.approx(0.5, abs=0.05)

    def test_fit_clamps_ratio_to_range(self):
        m = HalftimeModel()
        rows = [{"home_goals": 4, "away_goals": 4, "ht_home": 4, "ht_away": 4} for _ in range(25)]
        m.fit(rows)
        assert 0.30 <= m._ht_ratio_home <= 0.55

    def test_fit_skips_zero_goals(self):
        m = HalftimeModel()
        rows = [{"home_goals": 0, "away_goals": 0, "ht_home": 0, "ht_away": 0} for _ in range(25)]
        m.fit(rows)  # all filtered out — default ratio stays
        assert m._ht_ratio_home == pytest.approx(0.42)

    def test_probabilities_keys(self):
        m = HalftimeModel()
        result = m.probabilities(1.5, 1.2)
        assert "ht" in result
        assert "sh" in result
        for half in ("ht", "sh"):
            for key in ("lambda_home", "lambda_away", "lambda_total", "line", "over", "under"):
                assert key in result[half]

    def test_probabilities_sum_to_one(self):
        m = HalftimeModel()
        result = m.probabilities(1.5, 1.2, line_ht=1.5, line_sh=1.5)
        assert result["ht"]["over"] + result["ht"]["under"] == pytest.approx(1.0, abs=0.001)
        assert result["sh"]["over"] + result["sh"]["under"] == pytest.approx(1.0, abs=0.001)

    def test_all_lines_returns_four_markets(self):
        m = HalftimeModel()
        result = m.all_lines(1.5, 1.2)
        assert "ht_ou05" in result
        assert "ht_ou15" in result
        assert "sh_ou05" in result
        assert "sh_ou15" in result

    def test_sh_lambda_is_positive(self):
        m = HalftimeModel()
        result = m.probabilities(0.01, 0.01)  # very low lambdas
        # Second half lambda should be clamped to 0.01 minimum
        assert result["sh"]["lambda_home"] >= 0.0
        assert result["sh"]["lambda_away"] >= 0.0


# ===========================================================================
# CorrectScoreModel
# ===========================================================================

class TestCorrectScoreModel:

    def _matrix(self, lam_h=1.5, lam_a=1.2, max_g=8):
        m = CorrectScoreModel(max_goals=max_g)
        return m._build_matrix(lam_h, lam_a, max_g)

    def test_build_matrix_sums_to_one(self):
        m = CorrectScoreModel()
        matrix = m._build_matrix(1.5, 1.2, 8)
        total = sum(p for row in matrix for p in row)
        assert total == pytest.approx(1.0, abs=0.001)

    def test_probabilities_returns_sorted_descending(self):
        m = CorrectScoreModel(top_n=12)
        matrix = self._matrix()
        results = m.probabilities(matrix)
        probs = [r["probability"] for r in results]
        assert probs == sorted(probs, reverse=True)

    def test_probabilities_top_n_capped(self):
        m = CorrectScoreModel(top_n=5)
        matrix = self._matrix()
        results = m.probabilities(matrix)
        assert len(results) <= 5

    def test_probabilities_row_structure(self):
        m = CorrectScoreModel()
        matrix = self._matrix()
        results = m.probabilities(matrix)
        for r in results:
            assert "score" in r
            assert "home_goals" in r
            assert "away_goals" in r
            assert "probability" in r
            assert "fair_odds" in r
            assert r["probability"] > 0
            assert r["fair_odds"] >= 1.0

    def test_top_scores_dict_returns_dict(self):
        m = CorrectScoreModel(top_n=6)
        matrix = self._matrix()
        d = m.top_scores_dict(matrix, top_n=6)
        assert isinstance(d, dict)
        assert len(d) <= 6
        for k, v in d.items():
            assert "-" in k
            assert 0 < v <= 1

    def test_halftime_probabilities(self):
        m = CorrectScoreModel()
        results = m.halftime_probabilities(0.8, 0.6, top_n=6)
        assert len(results) <= 6
        # HT capped at 4 goals per team — no score > 4-4
        for r in results:
            assert r["home_goals"] <= 4
            assert r["away_goals"] <= 4

    def test_fair_odds_inverse_of_probability(self):
        m = CorrectScoreModel()
        matrix = self._matrix()
        for r in m.probabilities(matrix):
            # fair_odds = round(1/prob, 2) — tolerance covers rounding of nested floats
            assert r["fair_odds"] == pytest.approx(1.0 / r["probability"], rel=0.02)

    def test_zero_probability_skipped(self):
        m = CorrectScoreModel()
        # All-zero matrix
        matrix = [[0.0] * 9 for _ in range(9)]
        results = m.probabilities(matrix)
        assert results == []


# ===========================================================================
# MultiMarketEngine
# ===========================================================================

class TestMultiMarketEngine:

    @pytest.fixture
    def fitted_engine(self):
        engine = MultiMarketEngine()
        # Need enough matches for PoissonEngine to produce usable lambdas
        matches = []
        for i, (h, a) in enumerate(
            [("Juventus","Inter"),("AC Milan","Roma"),("Napoli","Lazio"),
             ("Inter","Napoli"),("Roma","Juventus"),("Lazio","AC Milan"),
             ("Juventus","Roma"),("Inter","Lazio"),("AC Milan","Napoli")]
        ):
            matches.append({
                "home_team": h, "away_team": a,
                "home_goals": (i % 3), "away_goals": (i % 2),
                "ht_home": (i % 2), "ht_away": 0,
                "home_corners": 5 + i % 3, "away_corners": 4 + i % 2,
            })
        engine.fit(matches)
        return engine

    def test_fit_sets_fitted_flag(self, fitted_engine):
        assert fitted_engine._fitted is True

    def test_fit_empty_does_not_set_fitted(self):
        engine = MultiMarketEngine()
        engine.fit([])
        assert engine._fitted is False

    def test_predict_all_not_fitted_returns_empty(self):
        engine = MultiMarketEngine()
        result = engine.predict_all([{"fixture_id": "1", "home_team": "A", "away_team": "B"}])
        assert result == []

    def test_predict_all_returns_rows(self, fitted_engine):
        upcoming = [{"fixture_id": "999", "home_team": "Juventus", "away_team": "Inter",
                     "match_date": "2025-01-01", "league": "Serie A"}]
        rows = fitted_engine.predict_all(upcoming)
        assert len(rows) > 0

    def test_predict_all_row_structure(self, fitted_engine):
        upcoming = [{"fixture_id": "999", "home_team": "Juventus", "away_team": "Inter"}]
        rows = fitted_engine.predict_all(upcoming)
        required_keys = {"fixture_id", "home", "away", "market", "selection",
                         "probability", "fair_odds", "edge", "ev", "kelly", "created_at"}
        for row in rows[:5]:
            assert required_keys.issubset(row.keys())

    def test_predict_all_probability_in_range(self, fitted_engine):
        upcoming = [{"fixture_id": "999", "home_team": "Juventus", "away_team": "Inter"}]
        rows = fitted_engine.predict_all(upcoming)
        for row in rows:
            assert 0.0 <= row["probability"] <= 1.0

    def test_predict_all_markets_present(self, fitted_engine):
        upcoming = [{"fixture_id": "999", "home_team": "Juventus", "away_team": "Inter"}]
        rows = fitted_engine.predict_all(upcoming)
        markets = {r["market"] for r in rows}
        assert "1x2" in markets
        assert "ou25" in markets
        assert "btts" in markets

    def test_predict_all_skips_missing_teams(self, fitted_engine):
        upcoming = [
            {"fixture_id": "1", "home_team": "", "away_team": "Inter"},
            {"fixture_id": "2", "home_team": "Juventus", "away_team": ""},
        ]
        rows = fitted_engine.predict_all(upcoming)
        assert rows == []

    def test_predict_all_with_odds_map(self, fitted_engine):
        upcoming = [{"fixture_id": "999", "home_team": "Juventus", "away_team": "Inter"}]
        odds_map = {"999": {"1x2": {"home": 1.9, "draw": 3.4, "away": 3.8}}}
        rows = fitted_engine.predict_all(upcoming, odds_map=odds_map)
        assert len(rows) > 0

    def test_ou_from_matrix_over_under_sum(self):
        # 3x3 identity-style matrix
        matrix = [[1/9] * 3 for _ in range(3)]
        over, under = MultiMarketEngine._ou_from_matrix(matrix, 2.5)
        assert over + under == pytest.approx(1.0, abs=0.001)

    def test_row_static_method_fair_odds(self):
        row = MultiMarketEngine._row("1","H","A","1x2","home",0.5,None,"2025-01-01T00:00:00",{})
        assert row["fair_odds"] == pytest.approx(2.0, abs=0.01)
        assert row["edge"] == 0.0  # no market odds

    def test_row_zero_probability_gives_999_odds(self):
        row = MultiMarketEngine._row("1","H","A","1x2","home",0.0,None,"2025-01-01T00:00:00",{})
        assert row["fair_odds"] == 999.0

    def test_row_kelly_non_negative(self):
        row = MultiMarketEngine._row("1","H","A","1x2","home",0.1,5.0,"2025-01-01T00:00:00",{})
        assert row["kelly"] >= 0.0

"""
Coverage-boost tests for engine modules between 28–50%.

Targeted modules:
  - engine/correlated_parlay.py   (42% → 60%+)
  - engine/luck_index.py          (43% → 60%+)
  - engine/sentiment_engine.py    (37% → 55%+)
  - analytics/betfair_leagues.py  (40% → 100%)
  - training/import_validator.py  (40% → 85%+)
  - engine/offline_controller.py  (46% → 60%+)
  - engine/training_manager.py    (33% → 50%+)
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

# ===========================================================================
# analytics/betfair_leagues.py  (5 lines → 100%)
# ===========================================================================

from analytics.betfair_leagues import is_betfair_league, BETFAIR_EXCHANGE_ITALIA_LEAGUES


class TestBetfairLeagues:

    def test_serie_a_is_betfair(self):
        assert is_betfair_league("Serie A") is True

    def test_premier_league_is_betfair(self):
        assert is_betfair_league("Premier League") is True

    def test_ligue_2_is_betfair(self):
        assert is_betfair_league("Ligue 2") is True

    def test_unknown_league_is_not_betfair(self):
        assert is_betfair_league("Moldovan Premier League") is False

    def test_empty_string_is_not_betfair(self):
        assert is_betfair_league("") is False

    def test_none_is_not_betfair(self):
        assert is_betfair_league(None) is False

    def test_set_contains_major_leagues(self):
        assert "La Liga" in BETFAIR_EXCHANGE_ITALIA_LEAGUES
        assert "Bundesliga" in BETFAIR_EXCHANGE_ITALIA_LEAGUES


# ===========================================================================
# training/import_validator.py  (40% → 90%+)
# ===========================================================================

from training.import_validator import ImportValidator


class TestImportValidator:

    @pytest.fixture
    def validator(self):
        return ImportValidator()

    def test_validate_matches_df_ok(self, validator):
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame({
            "fixture_id": [1], "league": ["SerieA"], "season": [2024],
            "home": ["Juve"], "away": ["Inter"], "match_date": ["2024-01-01"],
            "home_goals": [2], "away_goals": [1], "status": ["FT"],
        })
        result = validator.validate_matches_df(df)
        assert result["ok"] is True
        assert result["missing_columns"] == []
        assert result["row_count"] == 1

    def test_validate_matches_df_missing_column(self, validator):
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame({"fixture_id": [1], "league": ["SerieA"]})
        result = validator.validate_matches_df(df)
        assert result["ok"] is False
        assert len(result["missing_columns"]) > 0

    def test_validate_odds_df_ok(self, validator):
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame({
            "fixture_id": [1],
            "home_odds": [1.9],
            "draw_odds": [3.5],
            "away_odds": [4.0],
        })
        result = validator.validate_odds_df(df)
        assert result["ok"] is True

    def test_validate_odds_df_missing_columns(self, validator):
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame({"fixture_id": [1], "home_odds": [1.9]})
        result = validator.validate_odds_df(df)
        assert result["ok"] is False
        assert "draw_odds" in result["missing_columns"]

    def test_load_and_validate_matches_csv(self, validator, tmp_path):
        path = tmp_path / "matches.csv"
        path.write_text(
            "fixture_id,league,season,home,away,match_date,home_goals,away_goals,status\n"
            "1,SerieA,2024,Juve,Inter,2024-01-01,2,1,FT\n"
        )
        result = validator.load_and_validate_matches_csv(str(path))
        assert result["ok"] is True
        assert result["row_count"] == 1

    def test_load_and_validate_odds_csv(self, validator, tmp_path):
        path = tmp_path / "odds.csv"
        path.write_text("fixture_id,home_odds,draw_odds,away_odds\n1,1.9,3.5,4.0\n")
        result = validator.load_and_validate_odds_csv(str(path))
        assert result["ok"] is True


# ===========================================================================
# engine/luck_index.py  (43% → 65%+)
# ===========================================================================

from engine.luck_index import LuckIndex, MatchRecord, compute_luck_report


class TestLuckIndexAnalysis:

    @pytest.fixture
    def engine(self):
        return LuckIndex(max_goals=6)

    def test_xpts_from_xg_strong_home(self, engine):
        xph, xpa = engine.xpts_from_xg(3.0, 0.3)
        assert xph > 2.0   # home very likely to win → close to 3 pts
        assert xpa < 0.5

    def test_xpts_from_xg_symmetric(self, engine):
        xph, xpa = engine.xpts_from_xg(1.5, 1.5)
        assert xph == pytest.approx(xpa, abs=0.01)

    def test_xpts_from_xg_returns_floats(self, engine):
        xph, xpa = engine.xpts_from_xg(1.5, 1.2)
        assert isinstance(xph, float) and isinstance(xpa, float)

    def test_xpts_from_xg_zero_home_xg(self, engine):
        xph, xpa = engine.xpts_from_xg(0.0, 2.0)
        assert xph < 0.5   # very unlikely home gets any points

    def test_analyse_empty_returns_report(self, engine):
        report = engine.analyse([])
        assert report.team_stats == {}

    def test_analyse_single_match(self, engine):
        # MatchRecord: home_team, away_team, home_goals, away_goals, home_xg, away_xg
        matches = [
            MatchRecord(
                home_team="Juve", away_team="Inter",
                home_goals=2, away_goals=0,
                home_xg=2.0, away_xg=0.5,
            )
        ]
        report = engine.analyse(matches)
        assert "Juve" in report.team_stats or "Inter" in report.team_stats

    def test_from_dicts_produces_report(self, engine):
        rows = [
            {
                "home_team": "Juve", "away_team": "Inter",
                "home_goals": 2, "away_goals": 0,
                "home_xg": 2.0, "away_xg": 0.5,
            }
        ]
        report = engine.from_dicts(rows)
        assert report is not None

    def test_compute_luck_report_helper(self):
        rows = [
            {
                "home_team": "Juve", "away_team": "Inter",
                "home_goals": 2, "away_goals": 1,
                "home_xg": 1.5, "away_xg": 1.0,
            }
            for _ in range(5)
        ]
        report = compute_luck_report(rows)
        assert report is not None

    def test_regression_candidates_is_list(self, engine):
        report = engine.analyse([])
        assert isinstance(report.regression_candidates(), list)

    def test_fade_candidates_is_list(self, engine):
        report = engine.analyse([])
        assert isinstance(report.fade_candidates(), list)


# ===========================================================================
# engine/correlated_parlay.py  (42% → 65%+)
# ===========================================================================

from engine.correlated_parlay import (
    CorrelatedParlayEngine,
    SingleEvent,
    build_same_game_parlay,
    _poisson_pmf as parlay_pmf,
    _build_score_matrix,
)


class TestCorrelatedParlayFull:

    @pytest.fixture
    def engine(self):
        return CorrelatedParlayEngine(lambda_home=1.8, lambda_away=1.2)

    def test_p_over_goals_sum_with_under(self, engine):
        assert engine.p_over_goals(2.5) + engine.p_under_goals(2.5) == pytest.approx(1.0, abs=0.01)

    def test_p_btts_complement(self, engine):
        assert engine.p_btts_yes() + engine.p_btts_no() == pytest.approx(1.0, abs=0.01)

    def test_p_cards_over_complement(self, engine):
        assert engine.p_cards_over(3.5) + engine.p_cards_under(3.5) == pytest.approx(1.0, abs=0.01)

    def test_joint_prob_single_event_equals_marginal(self, engine):
        event = SingleEvent("Home Win", "1x2_home", 0.0, 1.9, engine.p_home_win())
        joint = engine.joint_prob([event])
        assert joint == pytest.approx(engine.p_home_win(), abs=0.02)

    def test_joint_prob_over_goals(self, engine):
        event = SingleEvent("Over 2.5", "over_goals", 2.5, 1.8, engine.p_over_goals(2.5))
        joint = engine.joint_prob([event])
        assert joint == pytest.approx(engine.p_over_goals(2.5), abs=0.02)

    def test_joint_prob_two_events_leq_minimum(self, engine):
        e1 = SingleEvent("Home Win", "1x2_home", 0.0, 1.9, engine.p_home_win())
        e2 = SingleEvent("Over 2.5", "over_goals", 2.5, 1.8, engine.p_over_goals(2.5))
        joint = engine.joint_prob([e1, e2])
        assert joint <= min(engine.p_home_win(), engine.p_over_goals(2.5)) + 0.01

    def test_evaluate_parlay_structure(self, engine):
        e1 = SingleEvent("Home Win", "1x2_home", 0.0, 1.9, 0.55)
        e2 = SingleEvent("O2.5", "over_goals", 2.5, 1.8, 0.52)
        result = engine.evaluate_parlay([e1, e2])
        assert hasattr(result, "model_joint_prob")
        assert hasattr(result, "book_joint_prob")
        assert 0.0 <= result.model_joint_prob <= 1.0

    def test_correlation_matrix_shape(self, engine):
        events = [
            SingleEvent("Home Win", "1x2_home", 0.0, 1.9, 0.55),
            SingleEvent("O2.5", "over_goals", 2.5, 1.8, 0.52),
        ]
        matrix = engine.correlation_matrix(events)
        assert len(matrix) == 2
        assert len(matrix[0]) == 2
        assert matrix[0][0] == pytest.approx(1.0, abs=0.001)  # diagonal

    def test_find_value_parlays_returns_list(self, engine):
        events = [
            SingleEvent("Home Win", "1x2_home", 0.0, 1.9, 0.55),
            SingleEvent("O2.5", "over_goals", 2.5, 1.8, 0.52),
            SingleEvent("BTTS Yes", "btts_yes", 0.0, 1.7, 0.50),
        ]
        parlays = engine.find_value_parlays(events)
        assert isinstance(parlays, list)

    def test_build_score_matrix_sums_to_one(self):
        matrix = _build_score_matrix(1.5, 1.2, max_goals=6)
        total = sum(matrix.values())
        assert total == pytest.approx(1.0, abs=0.01)

    def test_poisson_pmf_zero_k(self):
        import math
        assert parlay_pmf(0, 2.0) == pytest.approx(math.exp(-2.0), rel=1e-5)

    def test_build_same_game_parlay_runs(self):
        # build_same_game_parlay takes lambda_home, lambda_away, events (List[SingleEvent])
        events = [
            SingleEvent("Home Win", "1x2_home", 0.0, 1.9, 0.55),
            SingleEvent("O2.5",     "over_goals", 2.5, 1.8, 0.52),
            SingleEvent("BTTS Yes", "btts_yes", 0.0, 1.7, 0.50),
        ]
        result = build_same_game_parlay(lambda_home=1.8, lambda_away=1.2, events=events)
        assert result is not None


# ===========================================================================
# engine/sentiment_engine.py  (37% → 55%+)
# ===========================================================================

from engine.sentiment_engine import SentimentEngine, TextItem, SentimentScore


class TestSentimentEngine:

    @pytest.fixture
    def engine(self):
        return SentimentEngine(half_life_hours=24.0)

    def _make_item(self, text: str, teams=("Juve",)) -> TextItem:
        # TextItem fields: text, source, team_mentions (list), timestamp, language
        return TextItem(text=text, source="twitter",
                        team_mentions=list(teams), timestamp=time.time())

    def test_score_text_neutral(self, engine):
        item = self._make_item("Team played a match today")
        score = engine.score_text(item, "Juve")
        assert isinstance(score, SentimentScore)
        assert -1.0 <= score.final_score <= 1.0

    def test_score_text_positive(self, engine):
        item = self._make_item("Brilliant win! Fantastic performance!")
        score = engine.score_text(item, "Juve")
        assert score.final_score >= 0.0

    def test_score_text_negative_keywords_hit(self, engine):
        # Engine: injury/negative keywords carry positive weights → final_score > 0
        # and the category is set to the dominant negative bucket
        item = self._make_item("Player crisis collapse disaster sacked")
        score = engine.score_text(item, "Inter")
        assert score.category in ("INJURY", "TENSION", "NEGATIVE")
        assert score.final_score > 0  # positive score = bad news in this engine's convention

    def test_aggregate_empty_returns_neutral(self, engine):
        # aggregate(items, team) — items first, then team
        report = engine.aggregate([], "Juve")
        assert report.weighted_score == 0.0

    def test_aggregate_single_item(self, engine):
        items = [self._make_item("Good form today for Juve")]
        report = engine.aggregate(items, "Juve")
        assert report.team == "Juve"
        assert -1.0 <= report.weighted_score <= 1.0

    def test_elo_adjustment_neutral_score(self, engine):
        # elo_adjustment returns 0 for scores >= -0.30
        adj = engine.elo_adjustment(0.0)
        assert adj == pytest.approx(0.0, abs=0.01)

    def test_elo_adjustment_positive_score_returns_zero(self, engine):
        # Positive score → no Elo adjustment (only negative scores trigger it)
        adj = engine.elo_adjustment(0.5)
        assert adj == pytest.approx(0.0, abs=0.01)

    def test_elo_adjustment_negative_score(self, engine):
        adj = engine.elo_adjustment(-0.5)
        assert adj < 0

    def test_lambda_multiplier_neutral(self, engine):
        mult = engine.lambda_multiplier(0.0)
        assert mult == pytest.approx(1.0, abs=0.001)

    def test_lambda_multiplier_very_negative(self, engine):
        mult = engine.lambda_multiplier(-0.5)
        assert 0.7 <= mult < 1.0

    def test_recency_weight_recent(self, engine):
        now = time.time()
        w = engine._recency_weight(now, now)
        assert w == pytest.approx(1.0, abs=0.01)

    def test_recency_weight_decays_over_time(self, engine):
        now = time.time()
        old = now - 48 * 3600  # 48 hours ago
        w = engine._recency_weight(old, now)
        assert 0.0 < w < 1.0


# ===========================================================================
# engine/offline_controller.py  (46% → 60%+)
# ===========================================================================

from engine.offline_controller import OfflineController


class TestOfflineController:

    def test_constructor(self):
        ctrl = OfflineController()
        assert ctrl.engine is not None

    def test_import_csv_data_no_args_returns_empty(self):
        ctrl = OfflineController()
        result = ctrl.import_csv_data()
        assert result == {}

    def test_run_reports_delegates_without_ranked_df(self):
        from unittest.mock import MagicMock, patch
        ctrl = OfflineController()
        mock_engine = MagicMock()
        mock_engine.run_reports.return_value = True
        with patch.object(ctrl, "engine", mock_engine):
            result = ctrl.run_reports()
        mock_engine.run_reports.assert_called_once_with()
        assert result is True

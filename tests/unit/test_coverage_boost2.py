"""
Coverage-boost tests — second batch.

Targeted modules:
  - quant/providers/league_registry.py  (29% → 70%+)
  - engine/gaussian_copula.py           (35% → 60%+)
  - engine/training_manager.py          (33% → 55%+)
  - training/backtest_engine.py         (17% → 50%+)
"""

from __future__ import annotations

import pytest


# ===========================================================================
# quant/providers/league_registry.py  (29% → 70%+)
# ===========================================================================

from quant.providers.league_registry import name as league_name, all_known, _STATIC


class TestLeagueRegistry:

    def test_serie_a_known_statically(self):
        assert league_name(135) == "Serie A"

    def test_premier_league_known_statically(self):
        assert league_name(39) == "Premier League"

    def test_bundesliga_known_statically(self):
        assert league_name(78) == "Bundesliga"

    def test_ucl_known_statically(self):
        assert league_name(2) == "UEFA Champions League"

    def test_unknown_league_returns_fallback(self):
        result = league_name(999999)
        assert isinstance(result, str)
        assert len(result) > 0  # returns "League 999999" or similar

    def test_all_known_returns_dict(self):
        d = all_known()
        assert isinstance(d, dict)
        assert len(d) >= 50  # static table has 50+ major league entries

    def test_all_known_includes_main_leagues(self):
        d = all_known()
        assert 135 in d  # Serie A
        assert 39  in d  # Premier League

    def test_static_table_non_empty(self):
        assert len(_STATIC) > 0

    def test_name_with_no_db_no_api_uses_static(self):
        # Pass None for db_name and api_key — should resolve from static table
        result = league_name(62, db_name=None, api_key=None)
        assert result == "Ligue 2"

    def test_unknown_league_no_db_no_api_fallback(self):
        result = league_name(99999, db_name=None, api_key=None)
        assert isinstance(result, str)


# ===========================================================================
# engine/gaussian_copula.py  (35% → 60%+)
# ===========================================================================

from engine.gaussian_copula import (
    BetLeg,
    CopulaCorrelation,
    CopulaResult,
    GaussianCopulaEngine,
)
from engine.copula_math import (
    _normal_cdf,
    _normal_pdf,
    _normal_ppf,
    _make_positive_definite,
)


class TestGaussianHelpers:

    def test_normal_pdf_peak_at_zero(self):
        assert _normal_pdf(0.0) == pytest.approx(0.3989, abs=0.001)

    def test_normal_pdf_symmetric(self):
        assert _normal_pdf(-1.0) == pytest.approx(_normal_pdf(1.0), rel=1e-6)

    def test_normal_cdf_at_zero_is_half(self):
        assert _normal_cdf(0.0) == pytest.approx(0.5, abs=0.001)

    def test_normal_cdf_large_positive_near_one(self):
        assert _normal_cdf(10.0) > 0.999

    def test_normal_cdf_large_negative_near_zero(self):
        assert _normal_cdf(-10.0) < 0.001

    def test_normal_ppf_inverse_of_cdf(self):
        for p in (0.05, 0.25, 0.5, 0.75, 0.95):
            z = _normal_ppf(p)
            assert _normal_cdf(z) == pytest.approx(p, abs=0.001)

    def test_make_positive_definite_returns_matrix(self):
        # Near-PSD matrix
        m = [[1.0, 0.99], [0.99, 1.0]]
        result = _make_positive_definite(m)
        assert len(result) == 2
        assert len(result[0]) == 2
        # Diagonal must still be 1.0
        assert result[0][0] == pytest.approx(1.0, abs=0.01)


class TestBetLeg:

    def test_valid_leg(self):
        leg = BetLeg(name="Home Win", market_odds=2.0, model_prob=0.55)
        assert leg.bookmaker_implied == pytest.approx(0.5, abs=0.001)

    def test_invalid_prob_zero_raises(self):
        with pytest.raises(ValueError, match="model_prob"):
            BetLeg(name="X", market_odds=2.0, model_prob=0.0)

    def test_invalid_prob_one_raises(self):
        with pytest.raises(ValueError, match="model_prob"):
            BetLeg(name="X", market_odds=2.0, model_prob=1.0)

    def test_invalid_odds_one_raises(self):
        with pytest.raises(ValueError, match="market_odds"):
            BetLeg(name="X", market_odds=1.0, model_prob=0.5)

    def test_invalid_odds_below_one_raises(self):
        with pytest.raises(ValueError, match="market_odds"):
            BetLeg(name="X", market_odds=0.5, model_prob=0.5)


class TestCopulaCorrelation:

    def test_valid_correlation(self):
        corr = CopulaCorrelation(leg_i=0, leg_j=1, rho=0.5)
        assert corr.rho == 0.5

    def test_rho_above_one_raises(self):
        with pytest.raises(ValueError, match="rho"):
            CopulaCorrelation(leg_i=0, leg_j=1, rho=1.1)

    def test_rho_below_minus_one_raises(self):
        with pytest.raises(ValueError, match="rho"):
            CopulaCorrelation(leg_i=0, leg_j=1, rho=-1.1)

    def test_same_leg_raises(self):
        with pytest.raises(ValueError, match="distinct"):
            CopulaCorrelation(leg_i=0, leg_j=0, rho=0.5)


class TestGaussianCopulaEngine:

    @pytest.fixture
    def engine(self):
        # n_simulations must be >= 1000; use minimum for speed
        return GaussianCopulaEngine(n_simulations=1000, seed=42)

    @pytest.fixture
    def two_legs(self):
        return [
            BetLeg("Home Win",  market_odds=1.9, model_prob=0.55),
            BetLeg("Over 2.5",  market_odds=1.8, model_prob=0.52),
        ]

    def test_constructor_raises_below_1000_sims(self):
        with pytest.raises(ValueError):
            GaussianCopulaEngine(n_simulations=999)

    def test_constructor_raises_bad_default_corr(self):
        with pytest.raises(ValueError, match="default_correlation"):
            GaussianCopulaEngine(default_correlation=1.0)

    def test_evaluate_returns_copula_result(self, engine, two_legs):
        # evaluate requires event_types when no explicit correlation_matrix
        result = engine.evaluate(
            two_legs,
            event_types=["home_win", "over_goals"],
        )
        assert isinstance(result, CopulaResult)

    def test_evaluate_model_joint_in_range(self, engine, two_legs):
        result = engine.evaluate(two_legs, event_types=["home_win", "over_goals"])
        assert 0.0 <= result.model_joint_prob <= 1.0

    def test_evaluate_book_joint_in_range(self, engine, two_legs):
        result = engine.evaluate(two_legs, event_types=["home_win", "over_goals"])
        assert 0.0 < result.book_joint_prob <= 1.0

    def test_evaluate_with_explicit_matrix(self, engine, two_legs):
        corr_matrix = [[1.0, 0.3], [0.3, 1.0]]
        result = engine.evaluate(two_legs, correlation_matrix=corr_matrix)
        assert isinstance(result, CopulaResult)

    def test_evaluate_single_leg_raises(self, engine):
        legs = [BetLeg("Home Win", market_odds=1.9, model_prob=0.55)]
        with pytest.raises(ValueError, match="2 legs"):
            engine.evaluate(legs)

    def test_find_value_parlays_returns_list(self, engine, two_legs):
        results = engine.find_value_parlays(
            two_legs, event_types=["home_win", "over_goals"]
        )
        assert isinstance(results, list)

    def test_copula_result_str(self, engine, two_legs):
        result = engine.evaluate(two_legs, event_types=["home_win", "over_goals"])
        s = str(result)
        assert isinstance(s, str) and len(s) > 0


# ===========================================================================
# training/backtest_engine.py  (17% → 50%+)
# ===========================================================================

from training.backtest_engine import BacktestEngine


class TestBacktestEngine:

    @pytest.fixture
    def in_memory_conn(self):
        """In-memory SQLite with minimal schema for BacktestEngine."""
        import sqlite3
        conn = sqlite3.connect(":memory:")
        conn.executescript("""
            CREATE TABLE fixtures (
                fixture_id TEXT PRIMARY KEY,
                home_team TEXT, away_team TEXT,
                home_goals INTEGER, away_goals INTEGER,
                match_date TEXT, status TEXT,
                league_id INTEGER, league TEXT, season INTEGER
            );
            CREATE TABLE predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fixture_id TEXT,
                home_prob REAL, draw_prob REAL, away_prob REAL
            );
            CREATE TABLE odds_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fixture_id TEXT,
                home_odds REAL, draw_odds REAL, away_odds REAL,
                timestamp TEXT, market TEXT
            );
        """)
        return conn

    @pytest.fixture
    def engine(self):
        return BacktestEngine()

    def test_load_predictions_empty_db_returns_empty(self, engine, in_memory_conn):
        from unittest.mock import patch
        pytest.importorskip("pandas")
        with patch("training.backtest_engine.connect", return_value=in_memory_conn):
            result = engine.load_predictions_with_results()
        assert hasattr(result, "empty") and result.empty

    def test_run_empty_db_returns_empty(self, engine, in_memory_conn):
        from unittest.mock import patch
        pytest.importorskip("pandas")
        with patch("training.backtest_engine.connect", return_value=in_memory_conn):
            result = engine.run()
        assert hasattr(result, "empty") and result.empty

    def test_load_with_predictions_returns_dataframe(self, engine, in_memory_conn):
        pytest.importorskip("pandas")
        in_memory_conn.executescript("""
            INSERT INTO fixtures VALUES
                ('f1','Juve','Inter',2,1,'2024-01-01','FT',135,'SerieA',2024);
            INSERT INTO predictions(fixture_id,home_prob,draw_prob,away_prob)
                VALUES('f1',0.55,0.25,0.20);
        """)
        from unittest.mock import patch
        with patch("training.backtest_engine.connect", return_value=in_memory_conn):
            result = engine.load_predictions_with_results()
        assert len(result) == 1
        assert "actual_home_win" in result.columns


# ===========================================================================
# engine/training_manager.py  (33% → 55%+)
# ===========================================================================

from engine.training_manager import TrainingManager, _serialisable


class TestSerializable:

    def test_serialisable_int(self):
        assert _serialisable(42) == 42

    def test_serialisable_float(self):
        assert _serialisable(3.14) == pytest.approx(3.14)

    def test_serialisable_string(self):
        assert _serialisable("hello") == "hello"

    def test_serialisable_list(self):
        assert _serialisable([1, 2, 3]) == [1, 2, 3]

    def test_serialisable_dict(self):
        assert _serialisable({"a": 1}) == {"a": 1}

    def test_serialisable_none(self):
        assert _serialisable(None) is None

    def test_serialisable_unknown_type(self):
        class Foo:
            pass
        result = _serialisable(Foo())
        assert isinstance(result, str)  # falls back to str()


class TestTrainingManager:

    @pytest.fixture
    def manager(self):
        return TrainingManager(n_optuna_trials=1)

    def test_constructor_sets_attributes(self, manager):
        assert hasattr(manager, "dataset_builder")
        assert hasattr(manager, "automl")
        assert hasattr(manager, "backtest_engine")

    def test_dataset_summary_empty_df(self, manager):
        pd = pytest.importorskip("pandas")
        empty = pd.DataFrame()
        summary = manager.dataset_summary(empty)
        assert summary["rows"] == 0
        assert summary["features"] == []

    def test_dataset_summary_none(self, manager):
        summary = manager.dataset_summary(None)
        assert summary["rows"] == 0

    def test_dataset_summary_with_data(self, manager):
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame({
            "league": ["SerieA", "SerieA"],
            "match_date": ["2024-01-01", "2024-01-02"],
            "home_win_odds": [1.9, 2.1],
        })
        summary = manager.dataset_summary(df)
        assert summary["rows"] == 2
        assert "leagues" in summary

    def test_load_best_model_returns_none_when_absent(self, manager, tmp_path, monkeypatch):
        import training.local_automl as la
        monkeypatch.setattr(la, "MODEL_PATH", str(tmp_path / "model.pkl"))
        result = manager.load_best_model()
        assert result is None

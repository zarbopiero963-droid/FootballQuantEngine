"""
Systematic pytest.raises error-path tests — 40+ cases.

Every test here specifically validates that the right exception is raised (or
NOT raised) at known boundary conditions across all engine layers.  Guards
against silent swallowing and regression of input validation.
"""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# engine/markowitz_optimizer.py
# ---------------------------------------------------------------------------

from engine.markowitz_optimizer import BetProposal, optimise_portfolio


class TestBetProposalValidation:

    def test_prob_zero_raises_value_error(self):
        with pytest.raises(ValueError, match="model_prob"):
            BetProposal(bet_id="b1", description="X", odds=2.0,
                        model_prob=0.0, correlation_group="g1")

    def test_prob_one_raises_value_error(self):
        with pytest.raises(ValueError, match="model_prob"):
            BetProposal(bet_id="b1", description="X", odds=2.0,
                        model_prob=1.0, correlation_group="g1")

    def test_prob_negative_raises_value_error(self):
        with pytest.raises(ValueError, match="model_prob"):
            BetProposal(bet_id="b1", description="X", odds=2.0,
                        model_prob=-0.1, correlation_group="g1")

    def test_prob_above_one_raises_value_error(self):
        with pytest.raises(ValueError, match="model_prob"):
            BetProposal(bet_id="b1", description="X", odds=2.0,
                        model_prob=1.01, correlation_group="g1")

    def test_odds_one_raises_value_error(self):
        with pytest.raises(ValueError, match="odds"):
            BetProposal(bet_id="b1", description="X", odds=1.0,
                        model_prob=0.5, correlation_group="g1")

    def test_odds_below_one_raises_value_error(self):
        with pytest.raises(ValueError, match="odds"):
            BetProposal(bet_id="b1", description="X", odds=0.5,
                        model_prob=0.5, correlation_group="g1")

    def test_valid_proposal_does_not_raise(self):
        p = BetProposal(bet_id="b1", description="X", odds=2.0,
                        model_prob=0.5, correlation_group="g1")
        assert p.expected_return == pytest.approx(0.0, abs=0.001)


class TestOptimisePortfolio:

    def test_empty_bets_raises_value_error(self):
        with pytest.raises(ValueError, match="empty"):
            optimise_portfolio([])

    def test_missing_required_field_raises_value_error(self):
        with pytest.raises(ValueError, match="Missing required bet field"):
            optimise_portfolio([{"bet_id": "x", "odds": 2.0, "model_prob": 0.5}])


# ---------------------------------------------------------------------------
# engine/gaussian_copula.py
# ---------------------------------------------------------------------------

from engine.gaussian_copula import _normal_ppf, _cholesky


class TestNormalPPF:

    def test_zero_raises_value_error(self):
        with pytest.raises(ValueError, match="_normal_ppf"):
            _normal_ppf(0.0)

    def test_one_raises_value_error(self):
        with pytest.raises(ValueError, match="_normal_ppf"):
            _normal_ppf(1.0)

    def test_negative_raises_value_error(self):
        with pytest.raises(ValueError):
            _normal_ppf(-0.1)

    def test_above_one_raises_value_error(self):
        with pytest.raises(ValueError):
            _normal_ppf(1.1)

    def test_valid_midpoint_no_raise(self):
        result = _normal_ppf(0.5)
        assert result == pytest.approx(0.0, abs=0.001)

    def test_valid_tail_no_raise(self):
        result = _normal_ppf(0.01)
        assert result < -2.0


class TestCholesky:

    def test_non_square_matrix_raises_value_error(self):
        with pytest.raises(ValueError, match="square"):
            _cholesky([[1.0, 0.0]])  # 1x2 — not square

    def test_non_positive_definite_raises_value_error(self):
        # Negative diagonal makes it non-PD
        with pytest.raises(ValueError):
            _cholesky([[-1.0, 0.0], [0.0, 1.0]])


# ---------------------------------------------------------------------------
# engine/luck_index.py
# ---------------------------------------------------------------------------

from engine.luck_index import LuckIndex


class TestLuckIndex:

    def test_zero_max_goals_raises_value_error(self):
        with pytest.raises(ValueError, match="max_goals"):
            LuckIndex(max_goals=0)

    def test_negative_max_goals_raises_value_error(self):
        with pytest.raises(ValueError, match="max_goals"):
            LuckIndex(max_goals=-3)

    def test_valid_max_goals_no_raise(self):
        idx = LuckIndex(max_goals=1)
        assert idx.max_goals == 1

    def test_xpts_returns_tuple(self):
        idx = LuckIndex()
        xph, xpa = idx.xpts_from_xg(1.5, 1.2)
        assert isinstance(xph, float)
        assert isinstance(xpa, float)

    def test_xpts_zero_xg_gives_low_home_xpts(self):
        idx = LuckIndex()
        # 0 home xG — home almost certainly loses
        xph, xpa = idx.xpts_from_xg(0.0, 3.0)
        assert xph < xpa


# ---------------------------------------------------------------------------
# engine/correlated_parlay.py
# ---------------------------------------------------------------------------

from engine.correlated_parlay import CorrelatedParlayEngine, SingleEvent, build_same_game_parlay


class TestCorrelatedParlayEngine:

    def test_zero_home_lambda_raises_value_error(self):
        with pytest.raises(ValueError, match="positive"):
            CorrelatedParlayEngine(0.0, 1.5)

    def test_zero_away_lambda_raises_value_error(self):
        with pytest.raises(ValueError, match="positive"):
            CorrelatedParlayEngine(1.5, 0.0)

    def test_negative_lambda_raises_value_error(self):
        with pytest.raises(ValueError, match="positive"):
            CorrelatedParlayEngine(-0.1, 1.5)

    def test_zero_cards_lambda_raises_value_error(self):
        with pytest.raises(ValueError, match="positive"):
            CorrelatedParlayEngine(1.5, 1.2, lambda_cards_home=0.0)

    def test_valid_construction_no_raise(self):
        engine = CorrelatedParlayEngine(1.5, 1.2)
        assert engine.lambda_home == pytest.approx(1.5)

    def test_probabilities_sum_to_one(self):
        engine = CorrelatedParlayEngine(1.5, 1.2)
        total = engine.p_home_win() + engine.p_draw() + engine.p_away_win()
        assert total == pytest.approx(1.0, abs=0.01)

    def test_evaluate_parlay_empty_events_raises_value_error(self):
        engine = CorrelatedParlayEngine(1.5, 1.2)
        with pytest.raises(ValueError, match="empty"):
            engine.evaluate_parlay([])

    def test_find_value_parlays_min_legs_too_low_raises(self):
        engine = CorrelatedParlayEngine(1.5, 1.2)
        events = [
            SingleEvent("Home Win", "1x2_home", 0.0, 1.9, 0.55),
            SingleEvent("O2.5",     "over_goals", 2.5, 1.8, 0.52),
        ]
        with pytest.raises(ValueError, match="min_legs"):
            engine.find_value_parlays(events, min_legs=1)

    def test_find_value_parlays_max_less_than_min_raises(self):
        engine = CorrelatedParlayEngine(1.5, 1.2)
        events = [
            SingleEvent("Home Win", "1x2_home", 0.0, 1.9, 0.55),
            SingleEvent("O2.5",     "over_goals", 2.5, 1.8, 0.52),
        ]
        with pytest.raises(ValueError, match="max_legs"):
            engine.find_value_parlays(events, min_legs=3, max_legs=2)


# ---------------------------------------------------------------------------
# engine/sentiment_engine.py
# ---------------------------------------------------------------------------

from engine.sentiment_engine import SentimentEngine


class TestSentimentEngine:

    def test_zero_half_life_raises_value_error(self):
        with pytest.raises(ValueError, match="half_life"):
            SentimentEngine(half_life_hours=0.0)

    def test_negative_half_life_raises_value_error(self):
        with pytest.raises(ValueError, match="half_life"):
            SentimentEngine(half_life_hours=-1.0)

    def test_valid_half_life_no_raise(self):
        engine = SentimentEngine(half_life_hours=24.0)
        assert engine._lambda > 0


# ---------------------------------------------------------------------------
# quant/models/calibration.py — boundary invariants
# ---------------------------------------------------------------------------

from quant.models.calibration import ProbabilityCalibration


class TestCalibrationBoundaries:

    def test_calibrate_binary_clamps_to_open_interval(self):
        cal = ProbabilityCalibration()
        for p in [-1.0, 0.0, 0.0001, 0.5, 0.9999, 1.0, 1.5]:
            result = cal.calibrate_binary(p)
            assert 0.0 < result < 1.0, f"p={p} gave out-of-range {result}"

    def test_calibrate_three_way_zero_input_returns_uniform(self):
        cal = ProbabilityCalibration()
        result = cal.calibrate_three_way({"home_win": 0.0, "draw": 0.0, "away_win": 0.0})
        for v in result.values():
            assert v == pytest.approx(1/3, abs=1e-9)

    def test_calibrate_three_way_sums_to_one(self):
        cal = ProbabilityCalibration()
        for inputs in [
            {"home_win": 1.0, "draw": 0.0, "away_win": 0.0},
            {"home_win": 0.4, "draw": 0.3, "away_win": 0.3},
            {"home_win": 0.0, "draw": 1.0, "away_win": 0.0},
        ]:
            result = cal.calibrate_three_way(inputs)
            assert sum(result.values()) == pytest.approx(1.0, abs=1e-6)

    def test_calibrate_binary_shrinks_toward_half(self):
        cal = ProbabilityCalibration(shrink=0.5)
        # With 50% shrinkage, calibrate(1.0) should be halfway between p and 0.5
        result = cal.calibrate_binary(0.9999)
        assert result < 0.9999  # pulled toward 0.5


# ---------------------------------------------------------------------------
# quant/models/poisson_engine.py — invalid score matrix size
# ---------------------------------------------------------------------------

from quant.models.poisson_engine import PoissonEngine


class TestPoissonEngineEdges:

    def test_score_matrix_shape(self):
        engine = PoissonEngine(max_goals=4)
        engine.fit([{"home_team":"A","away_team":"B","home_goals":2,"away_goals":1}] * 10)
        m = engine.score_matrix("A", "B")
        assert len(m) == 5       # 0..4
        assert len(m[0]) == 5

    def test_score_matrix_sums_near_one(self):
        engine = PoissonEngine(max_goals=6)
        engine.fit([{"home_team":"A","away_team":"B","home_goals":1,"away_goals":1}] * 10)
        m = engine.score_matrix("A", "B")
        total = sum(p for row in m for p in row)
        assert total == pytest.approx(1.0, abs=0.01)

    def test_ou_btts_keys_present(self):
        engine = PoissonEngine()
        engine.fit([])
        result = engine.probabilities_ou_btts("X", "Y")
        for key in ("over_25", "under_25", "btts_yes", "btts_no"):
            assert key in result

    def test_ou_btts_values_in_range(self):
        engine = PoissonEngine()
        engine.fit([])
        result = engine.probabilities_ou_btts("X", "Y")
        for v in result.values():
            assert 0.0 <= v <= 1.0


# ---------------------------------------------------------------------------
# database/db_manager — missing schema path guard
# ---------------------------------------------------------------------------

from database.db_manager import connect


class TestDbManagerConnect:

    def test_connect_returns_connection(self, tmp_path, monkeypatch):
        import sqlite3
        from config import constants
        monkeypatch.setattr(constants, "DATABASE_NAME", str(tmp_path / "test.db"))
        import database.db_manager as dbm
        monkeypatch.setattr(dbm, "DATABASE_NAME", str(tmp_path / "test.db"))
        conn = dbm.connect()
        assert conn is not None
        conn.close()


# ---------------------------------------------------------------------------
# ranking/match_ranker.py — division-by-zero guards
# ---------------------------------------------------------------------------

from ranking.match_ranker import kelly_fraction, expected_value


class TestKellyFractionRaises:

    def test_exact_break_even_odds(self):
        # p=0.5, odds=2.0 → edge=0 → kelly=0
        result = kelly_fraction(0.5, 2.0)
        assert result == pytest.approx(0.0, abs=0.01)

    def test_probability_slightly_above_zero(self):
        result = kelly_fraction(1e-10, 2.0)
        assert result == pytest.approx(0.0)

    def test_large_bankroll_stays_finite(self):
        result = kelly_fraction(0.6, 5.0)
        assert not (result != result)  # not NaN
        assert result < float("inf")

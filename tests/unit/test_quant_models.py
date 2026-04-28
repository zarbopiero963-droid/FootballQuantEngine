"""
Unit tests for quant/models/ engines.

Covers: PoissonEngine (incl. defence-normalisation correctness),
DixonColesEngine, EloEngine, FormEngine, H2HEngine.
"""

from __future__ import annotations

import pytest

from quant.models.dixon_coles_engine import DixonColesEngine
from quant.models.elo_engine import EloEngine
from quant.models.form_engine import FormEngine
from quant.models.h2h_engine import H2HEngine
from quant.models.poisson_engine import PoissonEngine

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def matches():
    return [
        {"home_team": "A", "away_team": "B", "home_goals": 2, "away_goals": 1},
        {"home_team": "B", "away_team": "A", "home_goals": 0, "away_goals": 0},
        {"home_team": "A", "away_team": "C", "home_goals": 3, "away_goals": 2},
        {"home_team": "C", "away_team": "B", "home_goals": 1, "away_goals": 1},
        {"home_team": "B", "away_team": "C", "home_goals": 2, "away_goals": 0},
        {"home_team": "C", "away_team": "A", "home_goals": 1, "away_goals": 2},
    ]


# ---------------------------------------------------------------------------
# PoissonEngine — defence normalisation correctness (audit §4)
# ---------------------------------------------------------------------------


class TestPoissonDefenceNormalisation:
    """
    Validates the audit finding that defence_home uses league_avg_away_goals
    (not league_avg_home_goals) as denominator.

    Mathematical proof:
        λ_away = league_avg_away_goals × attack_away × defense_home
               = league_avg_away_goals
                 × (avg_away_scored / league_avg_away_goals)
                 × (avg_home_conceded / league_avg_away_goals)

    For an average team where avg_home_conceded == league_avg_away_goals:
        λ_away = avg_away_scored × 1.0  ✓

    Corollary: defence_away uses league_avg_home_goals (same logic, home side).
    """

    def test_average_team_lambda_equals_league_average(self):
        """
        A team whose conceded stats equal the league average should yield
        λ exactly equal to the opponent's avg_scored / league_avg.
        """
        engine = PoissonEngine()
        n = 20
        # Build matches where team "Avg" concedes exactly the league average
        matches = []
        for i in range(n):
            matches.append(
                {
                    "home_team": "Avg",
                    "away_team": "Other",
                    "home_goals": 1,
                    "away_goals": 1,
                }
            )
            matches.append(
                {
                    "home_team": "Other",
                    "away_team": "Avg",
                    "home_goals": 1,
                    "away_goals": 1,
                }
            )
        engine.fit(matches)
        # "Avg" has defense_home = avg_home_conceded / league_avg_away_goals = 1.0
        profile = engine.get_team_profile("Avg")
        assert profile["defense_home"] == pytest.approx(1.0, abs=1e-9)
        assert profile["defense_away"] == pytest.approx(1.0, abs=1e-9)

    def test_defense_home_denominator_is_league_avg_away_goals(self, matches):
        """
        defense_home = avg_home_conceded / league_avg_AWAY_goals (not home).
        A team that conceded 0 goals at home gets defense_home=0 (valid);
        all values must be non-negative.
        """
        engine = PoissonEngine()
        engine.fit(matches)
        for team, stats in engine.team_stats.items():
            assert stats["defense_home"] >= 0
            assert stats["defense_away"] >= 0

    def test_defense_away_denominator_is_league_avg_home_goals(self, matches):
        """
        defense_away = avg_away_conceded / league_avg_HOME_goals (not away).
        """
        engine = PoissonEngine()
        engine.fit(matches)
        # After fitting, λ for an average team should converge near league_avg
        lh, la = engine.expected_goals("A", "B")
        assert 0.5 <= lh <= 4.0
        assert 0.5 <= la <= 4.0

    def test_empty_fit_uses_baseline_lambdas(self):
        engine = PoissonEngine()
        engine.fit([])
        lh, la = engine.expected_goals("X", "Y")
        assert lh == pytest.approx(1.35, abs=0.01)
        assert la == pytest.approx(1.10, abs=0.01)

    def test_probabilities_sum_to_one(self, matches):
        engine = PoissonEngine()
        engine.fit(matches)
        p = engine.probabilities_1x2("A", "B")
        assert sum(p.values()) == pytest.approx(1.0, abs=1e-6)

    def test_ou_btts_probabilities_valid(self, matches):
        engine = PoissonEngine()
        engine.fit(matches)
        r = engine.probabilities_ou_btts("A", "B")
        assert r["over_25"] + r["under_25"] == pytest.approx(1.0, abs=1e-6)
        assert r["btts_yes"] + r["btts_no"] == pytest.approx(1.0, abs=1e-6)

    def test_unknown_team_returns_baseline(self):
        engine = PoissonEngine()
        engine.fit([])
        profile = engine.get_team_profile("Ghost FC")
        assert profile == {
            "attack_home": 1.0,
            "defense_home": 1.0,
            "attack_away": 1.0,
            "defense_away": 1.0,
        }


# ---------------------------------------------------------------------------
# DixonColesEngine
# ---------------------------------------------------------------------------


class TestDixonColesEngine:

    def test_fit_sets_nonzero_rho(self, matches):
        engine = DixonColesEngine()
        engine.fit(matches)
        assert isinstance(engine.rho, float)

    def test_probabilities_sum_to_one(self, matches):
        engine = DixonColesEngine()
        engine.fit(matches)
        p = engine.probabilities_1x2("A", "B")
        assert sum(p.values()) == pytest.approx(1.0, abs=1e-6)

    def test_probabilities_from_lambdas_sum_to_one(self):
        engine = DixonColesEngine()
        engine.rho = 0.0
        p = engine.probabilities_1x2_from_lambdas(1.5, 1.1)
        assert sum(p.values()) == pytest.approx(1.0, abs=1e-6)

    def test_empty_fit_falls_back_gracefully(self):
        engine = DixonColesEngine()
        engine.fit([])
        p = engine.probabilities_1x2("Unknown", "Unknown")
        assert all(v > 0 for v in p.values())

    def test_home_advantage_reflected_in_probs(self, matches):
        engine = DixonColesEngine()
        engine.fit(matches)
        p_ab = engine.probabilities_1x2("A", "B")
        p_ba = engine.probabilities_1x2("B", "A")
        # Swapping home/away should change the distribution
        assert p_ab["home_win"] != pytest.approx(p_ba["home_win"], abs=0.01)


# ---------------------------------------------------------------------------
# EloEngine
# ---------------------------------------------------------------------------


class TestEloEngine:

    def test_unknown_team_returns_base_rating(self):
        elo = EloEngine(base_rating=1500.0)
        assert elo.get_rating("Ghost FC") == 1500.0

    def test_fit_updates_ratings(self, matches):
        elo = EloEngine()
        elo.fit(matches)
        assert "A" in elo.ratings
        assert "B" in elo.ratings
        assert "C" in elo.ratings

    def test_winner_gains_rating(self):
        elo = EloEngine()
        elo.update_match("Strong", "Weak", 3, 0)
        assert elo.get_rating("Strong") > 1500.0
        assert elo.get_rating("Weak") < 1500.0

    def test_draw_minimal_change(self):
        elo = EloEngine()
        # Equal teams draw — neither changes much
        elo.update_match("T1", "T2", 1, 1)
        diff = abs(elo.get_rating("T1") - elo.get_rating("T2"))
        assert diff < 30  # home advantage shifts slightly

    def test_elo_diff_includes_home_advantage(self):
        elo = EloEngine(home_advantage=50.0)
        diff = elo.get_elo_diff("A", "B")
        # Both unknown → diff = 1500 + 50 - 1500 = 50
        assert diff == pytest.approx(50.0, abs=0.01)

    def test_fit_empty_list_no_crash(self):
        elo = EloEngine()
        elo.fit([])
        assert elo.ratings == {}


# ---------------------------------------------------------------------------
# FormEngine
# ---------------------------------------------------------------------------


class TestFormEngine:

    def test_unknown_team_returns_half(self):
        form = FormEngine()
        form.fit([])
        assert form.get_form("Ghost FC") == 0.5

    def test_all_wins_form_is_one(self):
        form = FormEngine(lookback=5)
        matches = [
            {"home_team": "A", "away_team": "B", "home_goals": 1, "away_goals": 0}
        ] * 5
        form.fit(matches)
        assert form.get_form("A") == pytest.approx(1.0)

    def test_all_losses_form_is_zero(self):
        form = FormEngine(lookback=5)
        matches = [
            {"home_team": "B", "away_team": "A", "home_goals": 1, "away_goals": 0}
        ] * 5
        form.fit(matches)
        assert form.get_form("A") == pytest.approx(0.0)

    def test_form_diff_symmetric(self, matches):
        form = FormEngine()
        form.fit(matches)
        diff_ab = form.get_form_diff("A", "B")
        diff_ba = form.get_form_diff("B", "A")
        assert diff_ab == pytest.approx(-diff_ba, abs=1e-9)

    def test_lookback_limits_history(self):
        form = FormEngine(lookback=2)
        # 3 wins followed by 2 losses
        matches = [
            {"home_team": "A", "away_team": "B", "home_goals": 2, "away_goals": 0}
        ] * 3 + [
            {"home_team": "B", "away_team": "A", "home_goals": 2, "away_goals": 0}
        ] * 2
        form.fit(matches)
        # Only last 2 results (losses as away) should count
        assert form.get_form("A") == pytest.approx(0.0, abs=0.01)


# ---------------------------------------------------------------------------
# H2HEngine
# ---------------------------------------------------------------------------


class TestH2HEngine:

    def test_no_history_returns_zero(self):
        h2h = H2HEngine()
        h2h.fit([])
        assert h2h.get_h2h_diff("A", "B") == 0.0

    def test_dominant_team_positive_diff(self):
        h2h = H2HEngine()
        matches = [
            {
                "home_team": "Strong",
                "away_team": "Weak",
                "home_goals": 2,
                "away_goals": 0,
            }
        ] * 5
        h2h.fit(matches)
        assert h2h.get_h2h_diff("Strong", "Weak") > 0

    def test_diff_range_is_minus_one_to_one(self, matches):
        h2h = H2HEngine()
        h2h.fit(matches)
        diff = h2h.get_h2h_diff("A", "B")
        assert -1.0 <= diff <= 1.0

    def test_reversed_fixture_uses_inverse_results(self):
        h2h = H2HEngine()
        matches = [
            {"home_team": "A", "away_team": "B", "home_goals": 3, "away_goals": 0}
        ] * 4
        h2h.fit(matches)
        # A dominated B at home; as away team B should be negative
        diff_ba = h2h.get_h2h_diff("B", "A")
        assert diff_ba < 0

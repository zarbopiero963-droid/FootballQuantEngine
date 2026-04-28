"""
Tests for simulation.monte_carlo (MonteCarloSimulator).
Covers simulate(), simulate_av(), and simulate_exact().
"""

from __future__ import annotations

import pytest

from simulation.monte_carlo import MonteCarloSimulator


@pytest.fixture
def sim():
    return MonteCarloSimulator()


class TestProbabilitiesSum:
    """All three methods must return probabilities that sum to 1."""

    def test_simulate_sums_to_one(self, sim):
        r = sim.simulate(1.5, 1.2, simulations=5000)
        assert abs(r["home_win"] + r["draw"] + r["away_win"] - 1.0) < 0.02

    def test_simulate_av_sums_to_one(self, sim):
        r = sim.simulate_av(1.5, 1.2, simulations=5000)
        assert abs(r["home_win"] + r["draw"] + r["away_win"] - 1.0) < 0.02

    def test_simulate_exact_sums_to_one(self, sim):
        r = sim.simulate_exact(1.5, 1.2)
        assert abs(r["home_win"] + r["draw"] + r["away_win"] - 1.0) < 1e-6


class TestSimulateExactDirectionality:
    """simulate_exact() should correctly reflect which team has the advantage."""

    def test_high_home_lambda_home_wins_more(self, sim):
        r = sim.simulate_exact(3.0, 0.5)
        assert r["home_win"] > r["away_win"]

    def test_high_away_lambda_away_wins_more(self, sim):
        r = sim.simulate_exact(0.5, 3.0)
        assert r["away_win"] > r["home_win"]

    def test_equal_lambdas_draw_plausible(self, sim):
        r = sim.simulate_exact(1.0, 1.0)
        assert 0.20 < r["draw"] < 0.50

    def test_all_probs_non_negative(self, sim):
        r = sim.simulate_exact(1.5, 1.2)
        assert r["home_win"] >= 0 and r["draw"] >= 0 and r["away_win"] >= 0


class TestEdgeCases:
    def test_zero_away_lambda(self, sim):
        r = sim.simulate_exact(1.5, 0.0)
        assert r["home_win"] > 0.5

    def test_very_small_lambdas(self, sim):
        r = sim.simulate_exact(0.01, 0.01)
        assert abs(r["home_win"] + r["draw"] + r["away_win"] - 1.0) < 1e-6

    def test_simulate_av_direction_correct(self, sim):
        r = sim.simulate_av(2.5, 0.5, simulations=10000)
        assert r["home_win"] > r["away_win"]

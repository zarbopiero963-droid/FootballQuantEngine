"""
Unit tests for simulation/monte_carlo_advanced.py and
analytics/market_inefficiency_scanner.py — coverage targets.
"""

from __future__ import annotations

import time

import pytest

from simulation.monte_carlo_advanced import (
    MCResult,
    MonteCarloAdvanced,
    _accumulate,
    _bivariate_poisson_sample,
    _poisson_sample,
)
from analytics.market_inefficiency_scanner import (
    MarketInefficiencyScanner,
    _classify_type,
    _cohen_h,
    _decay_weight,
    _edge_pct,
    _implied_prob,
    _remove_vig,
    _z_score,
)

# ===========================================================================
# MonteCarloAdvanced
# ===========================================================================


class TestMonteCarloAdvanced:

    @pytest.fixture
    def sim(self):
        return MonteCarloAdvanced()

    def test_simulate_returns_mc_result(self, sim):
        result = sim.simulate(1.5, 1.2, simulations=1_000, seed=0)
        assert isinstance(result, MCResult)

    def test_probabilities_sum_to_one(self, sim):
        result = sim.simulate(1.5, 1.2, simulations=2_000, seed=1)
        total = result.home_win + result.draw + result.away_win
        assert total == pytest.approx(1.0, abs=0.02)

    def test_over_under_complement(self, sim):
        result = sim.simulate(1.5, 1.2, simulations=2_000, seed=2)
        assert result.over_25 + result.under_25 == pytest.approx(1.0, abs=0.001)
        assert result.over_15 + result.under_15 == pytest.approx(1.0, abs=0.001)
        assert result.over_35 + result.under_35 == pytest.approx(1.0, abs=0.001)

    def test_btts_complement(self, sim):
        result = sim.simulate(1.5, 1.2, simulations=2_000, seed=3)
        assert result.btts_yes + result.btts_no == pytest.approx(1.0, abs=0.001)

    def test_simulations_count_recorded(self, sim):
        result = sim.simulate(1.5, 1.2, simulations=1_000, seed=4)
        assert result.simulations == 1_000

    def test_lambdas_recorded(self, sim):
        result = sim.simulate(2.0, 1.0, simulations=1_000, seed=5)
        assert result.lambda_home == pytest.approx(2.0)
        assert result.lambda_away == pytest.approx(1.0)

    def test_strong_home_team_wins_more(self, sim):
        strong = sim.simulate(3.0, 0.5, simulations=5_000, seed=6)
        weak = sim.simulate(0.5, 3.0, simulations=5_000, seed=6)
        assert strong.home_win > 0.5
        assert weak.away_win > 0.5

    def test_score_matrix_non_empty(self, sim):
        result = sim.simulate(1.5, 1.2, simulations=2_000, seed=7)
        assert len(result.score_matrix) > 0

    def test_score_matrix_probabilities_positive(self, sim):
        result = sim.simulate(1.5, 1.2, simulations=2_000, seed=8)
        for k, v in result.score_matrix.items():
            assert v >= 0

    def test_ci_home_ordered(self, sim):
        result = sim.simulate(1.5, 1.2, simulations=2_000, seed=9)
        lo, hi = result.ci_home
        assert lo <= hi

    def test_to_dict_complete_keys(self, sim):
        result = sim.simulate(1.5, 1.2, simulations=1_000, seed=10)
        d = result.to_dict()
        for key in (
            "home_win",
            "draw",
            "away_win",
            "over_25",
            "under_25",
            "btts_yes",
            "btts_no",
            "simulations",
            "converged",
            "ci_home",
            "ci_draw",
            "ci_away",
        ):
            assert key in d

    def test_lambda_shared_zero(self, sim):
        result = sim.simulate(1.5, 1.2, lambda_shared=0.0, simulations=1_000, seed=11)
        assert result.lambda_shared == pytest.approx(0.0)

    def test_very_low_lambdas_valid(self, sim):
        result = sim.simulate(0.1, 0.1, simulations=1_000, seed=12)
        total = result.home_win + result.draw + result.away_win
        assert total == pytest.approx(1.0, abs=0.05)

    def test_high_lambda_high_over25(self, sim):
        result = sim.simulate(4.0, 3.0, simulations=5_000, seed=13)
        assert result.over_25 > 0.7

    def test_seed_reproducibility(self, sim):
        r1 = sim.simulate(1.5, 1.2, simulations=1_000, seed=42)
        r2 = sim.simulate(1.5, 1.2, simulations=1_000, seed=42)
        assert r1.home_win == pytest.approx(r2.home_win, abs=0.001)

    def test_asian_handicap_m05_leq_m10(self, sim):
        result = sim.simulate(2.0, 1.0, simulations=5_000, seed=14)
        # P(home covers AH -0.5) >= P(home covers AH -1.0)
        assert result.ah_home_m05 >= result.ah_home_m10


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


class TestMCHelpers:

    def test_poisson_sample_zero_lambda(self):
        import random

        rng = random.Random(0)
        assert _poisson_sample(0.0, rng) == 0

    def test_poisson_sample_typical(self):
        import random

        rng = random.Random(42)
        samples = [_poisson_sample(2.0, rng) for _ in range(1000)]
        mean = sum(samples) / len(samples)
        assert 1.5 <= mean <= 2.5

    def test_bivariate_poisson_sample_nonneg(self):
        import random

        rng = random.Random(0)
        for _ in range(50):
            x, y = _bivariate_poisson_sample(1.5, 1.2, 0.1, rng)
            assert x >= 0 and y >= 0

    def test_accumulate_home_win(self):
        acc = {
            "hw": 0,
            "dr": 0,
            "aw": 0,
            "o15": 0,
            "o25": 0,
            "o35": 0,
            "o45": 0,
            "btts": 0,
            "ah05": 0,
            "ah10": 0,
            "ah15": 0,
            "ah20": 0,
            "score": {},
        }
        _accumulate(3, 0, acc)
        assert acc["hw"] == 1
        assert acc["dr"] == 0
        assert acc["aw"] == 0
        assert acc["o25"] == 1
        assert acc["btts"] == 0
        assert acc["ah05"] == 1
        assert acc["ah10"] == 1
        assert acc["ah20"] == 1

    def test_accumulate_draw(self):
        acc = {
            "hw": 0,
            "dr": 0,
            "aw": 0,
            "o15": 0,
            "o25": 0,
            "o35": 0,
            "o45": 0,
            "btts": 0,
            "ah05": 0,
            "ah10": 0,
            "ah15": 0,
            "ah20": 0,
            "score": {},
        }
        _accumulate(1, 1, acc)
        assert acc["dr"] == 1
        assert acc["btts"] == 1
        assert acc["o15"] == 1

    def test_accumulate_away_win(self):
        acc = {
            "hw": 0,
            "dr": 0,
            "aw": 0,
            "o15": 0,
            "o25": 0,
            "o35": 0,
            "o45": 0,
            "btts": 0,
            "ah05": 0,
            "ah10": 0,
            "ah15": 0,
            "ah20": 0,
            "score": {},
        }
        _accumulate(0, 2, acc)
        assert acc["aw"] == 1
        assert acc["ah05"] == 0


# ===========================================================================
# MarketInefficiencyScanner — statistical helpers
# ===========================================================================


class TestMISHelpers:

    def test_implied_prob_typical(self):
        assert _implied_prob(2.0) == pytest.approx(0.5, abs=0.001)

    def test_implied_prob_clamped(self):
        # odds = 0.5 → clamped to 1.01 → implied = 1/1.01 ≈ 0.99
        result = _implied_prob(0.5)
        assert result < 1.0

    def test_remove_vig_sums_to_one(self):
        ph, pd, pa = _remove_vig(2.0, 3.5, 4.0)
        assert ph + pd + pa == pytest.approx(1.0, abs=0.001)

    def test_remove_vig_zero_returns_uniform(self):
        ph, pd, pa = _remove_vig(0.0, 0.0, 0.0)
        assert ph == pytest.approx(1 / 3, abs=0.001)

    def test_z_score_no_difference_zero(self):
        z = _z_score(0.5, 0.5, 100)
        assert z == pytest.approx(0.0, abs=0.001)

    def test_z_score_large_n_large_z(self):
        z = abs(_z_score(0.6, 0.4, 10_000))
        assert z > 10

    def test_cohen_h_identical(self):
        assert _cohen_h(0.5, 0.5) == pytest.approx(0.0, abs=0.001)

    def test_cohen_h_large_gap(self):
        h = abs(_cohen_h(0.8, 0.2))
        assert h > 0.5

    def test_decay_weight_recent(self):
        now = time.time()
        w = _decay_weight(now, now)
        assert w == pytest.approx(1.0, abs=0.001)

    def test_decay_weight_old_observation(self):
        now = time.time()
        week_ago = now - 7 * 24 * 3600
        w = _decay_weight(week_ago, now)
        assert 0.0 < w < 1.0

    def test_edge_pct_positive(self):
        assert _edge_pct(0.6, 0.5) == pytest.approx(20.0, abs=0.1)

    def test_edge_pct_zero_implied(self):
        assert _edge_pct(0.5, 0.0) == 0.0

    def test_classify_type_draw_aversion(self):
        assert _classify_type("draw", 0.30, 0.20, 0.10) == "TYPE_C"

    def test_classify_type_draw_no_aversion(self):
        assert _classify_type("draw", 0.25, 0.24, 0.01) == "TYPE_D"

    def test_classify_type_home_favourite_underpriced(self):
        t = _classify_type("home", 0.55, 0.52, 0.03)
        assert t == "TYPE_A"

    def test_classify_type_home_longshot_overpriced(self):
        t = _classify_type("home", 0.20, 0.25, -0.05)
        assert t == "TYPE_A"

    def test_classify_type_home_advantage(self):
        t = _classify_type("home", 0.40, 0.42, -0.02)
        assert t == "TYPE_B"

    def test_classify_type_away_general(self):
        t = _classify_type("away", 0.30, 0.30, 0.00)
        assert t in ("TYPE_A", "TYPE_D")


# ===========================================================================
# MarketInefficiencyScanner — scan() and report()
# ===========================================================================


class TestMarketInefficiencyScanner:

    @pytest.fixture
    def scanner(self):
        return MarketInefficiencyScanner()

    def _pred(
        self,
        home_odds=1.9,
        draw_odds=3.4,
        away_odds=4.0,
        p_home=0.55,
        p_draw=0.25,
        p_away=0.20,
        league="SerieA",
    ):
        return {
            "fixture_id": "test_1",
            "league": league,
            "home_win": p_home,
            "draw": p_draw,
            "away_win": p_away,
            "home_odds": home_odds,
            "draw_odds": draw_odds,
            "away_odds": away_odds,
            "timestamp": time.time(),
            "n_history": 50,
        }

    def test_scan_empty_list(self, scanner):
        assert scanner.scan([]) == []

    def test_scan_returns_same_length(self, scanner):
        preds = [self._pred() for _ in range(5)]
        results = scanner.scan(preds)
        assert len(results) == 5

    def test_scan_adds_inefficiency_score(self, scanner):
        results = scanner.scan([self._pred()])
        assert "inefficiency_score" in results[0]
        assert results[0]["inefficiency_score"] >= 0

    def test_scan_invalid_odds_returns_none_type(self, scanner):
        pred = self._pred(home_odds=0.5, draw_odds=0.5, away_odds=0.5)
        results = scanner.scan([pred])
        assert results[0]["inefficiency_type"] == "NONE"

    def test_scan_strong_home_favourite_detected(self, scanner):
        # Model: 70% home, book: 1.3 home odds (implied ~77%) → home underpriced
        pred = self._pred(
            home_odds=1.3,
            draw_odds=4.5,
            away_odds=8.0,
            p_home=0.70,
            p_draw=0.20,
            p_away=0.10,
        )
        results = scanner.scan([pred])
        assert results[0]["inefficiency_score"] >= 0

    def test_scan_preserves_original_keys(self, scanner):
        pred = self._pred()
        pred["custom_field"] = "hello"
        results = scanner.scan([pred])
        assert results[0]["custom_field"] == "hello"

    def test_report_empty_history(self, scanner):
        r = scanner.report()
        assert r["n_observations"] == 0
        assert r["leagues"] == {}

    def test_report_after_scan(self, scanner):
        preds = [self._pred(league="SerieA") for _ in range(5)]
        results = scanner.scan(preds)
        r = scanner.report(results)
        assert "n_observations" in r

    def test_report_groups_by_league(self, scanner):
        preds = [self._pred(league="SerieA") for _ in range(3)] + [
            self._pred(league="PremierLeague") for _ in range(3)
        ]
        results = scanner.scan(preds)
        scanner.scan(preds)  # populate history
        r = scanner.report(results)
        if r["leagues"]:
            leagues = set(r["leagues"].keys())
            assert len(leagues) >= 1

    def test_scan_multiple_predictions(self, scanner):
        preds = [
            self._pred(p_home=0.6, p_draw=0.25, p_away=0.15, league="L1"),
            self._pred(p_home=0.3, p_draw=0.35, p_away=0.35, league="L2"),
            self._pred(home_odds=0.5),  # bad odds → NONE
        ]
        results = scanner.scan(preds)
        assert len(results) == 3

    def test_scan_with_dataframe(self, scanner):
        pd = pytest.importorskip("pandas")
        preds = [self._pred() for _ in range(3)]
        df = pd.DataFrame(preds)
        results = scanner.scan(df)
        assert hasattr(results, "shape")  # still a DataFrame
        assert results.shape[0] == 3

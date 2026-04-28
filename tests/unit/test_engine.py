"""
Unit tests for engine/markowitz_optimizer.py — BetProposal validation,
portfolio optimization, PSD regularisation.
"""

from __future__ import annotations

import pytest

from engine.markowitz_optimizer import (
    BetProposal,
    MarkowitzOptimizer,
    optimise_portfolio,
)

# ---------------------------------------------------------------------------
# BetProposal — validation & computed fields
# ---------------------------------------------------------------------------


class TestBetProposal:

    def _make(self, odds=2.10, model_prob=0.54, group="league_a"):
        return BetProposal(
            bet_id="b1",
            description="Test bet",
            odds=odds,
            model_prob=model_prob,
            correlation_group=group,
        )

    def test_valid_proposal_created(self):
        bp = self._make()
        assert bp.odds == 2.10
        assert bp.model_prob == 0.54

    def test_expected_return_formula(self):
        bp = self._make(odds=2.0, model_prob=0.6)
        assert bp.expected_return == pytest.approx(0.6 * 2.0 - 1.0, abs=1e-9)

    def test_variance_formula(self):
        bp = self._make(odds=2.0, model_prob=0.6)
        expected_var = 0.6 * 0.4 * (2.0**2)
        assert bp.variance == pytest.approx(expected_var, abs=1e-9)

    def test_kelly_fraction_positive_ev(self):
        bp = self._make(odds=2.5, model_prob=0.55)
        assert bp.kelly_fraction > 0.0

    def test_kelly_fraction_negative_ev_is_zero(self):
        # model_prob=0.3, odds=2.0 → EV = 0.3*2.0 - 1 = -0.4 → Kelly=0
        bp = self._make(odds=2.0, model_prob=0.3)
        assert bp.kelly_fraction == pytest.approx(0.0)

    def test_edge_pct_positive_for_value_bet(self):
        bp = self._make(odds=2.0, model_prob=0.6)
        # implied_prob = 0.5, model=0.6 → edge = 10%
        assert bp.edge_pct == pytest.approx(10.0, abs=0.01)

    def test_model_prob_zero_raises(self):
        with pytest.raises(ValueError, match="model_prob"):
            BetProposal(
                bet_id="x",
                description="x",
                odds=2.0,
                model_prob=0.0,
                correlation_group="g",
            )

    def test_model_prob_one_raises(self):
        with pytest.raises(ValueError, match="model_prob"):
            BetProposal(
                bet_id="x",
                description="x",
                odds=2.0,
                model_prob=1.0,
                correlation_group="g",
            )

    def test_model_prob_negative_raises(self):
        with pytest.raises(ValueError):
            BetProposal(
                bet_id="x",
                description="x",
                odds=2.0,
                model_prob=-0.1,
                correlation_group="g",
            )

    def test_odds_at_one_raises(self):
        with pytest.raises(ValueError, match="odds"):
            BetProposal(
                bet_id="x",
                description="x",
                odds=1.0,
                model_prob=0.5,
                correlation_group="g",
            )

    def test_odds_below_one_raises(self):
        with pytest.raises(ValueError, match="odds"):
            BetProposal(
                bet_id="x",
                description="x",
                odds=0.5,
                model_prob=0.5,
                correlation_group="g",
            )


# ---------------------------------------------------------------------------
# MarkowitzOptimizer — kelly_naive
# ---------------------------------------------------------------------------


def _make_bets(n: int = 3):
    bets = []
    for i in range(n):
        bets.append(
            BetProposal(
                bet_id=f"b{i}",
                description=f"Bet {i}",
                odds=2.0 + i * 0.3,
                model_prob=0.55 + i * 0.02,
                correlation_group="group_a",
                same_match_group="" if i < 2 else "match_1",
            )
        )
    return bets


class TestMarkowitzOptimizerKellyNaive:

    def test_kelly_naive_returns_allocation(self):
        opt = MarkowitzOptimizer()
        allocation = opt.kelly_naive(_make_bets(3))
        assert allocation is not None

    def test_kelly_naive_weights_nonnegative(self):
        opt = MarkowitzOptimizer()
        allocation = opt.kelly_naive(_make_bets(3))
        for w in allocation.weights:
            assert w >= 0.0

    def test_kelly_naive_total_weight_at_most_one(self):
        opt = MarkowitzOptimizer(bankroll_fraction=0.5)
        allocation = opt.kelly_naive(_make_bets(3))
        assert sum(allocation.weights) <= 0.5 + 1e-9

    def test_single_bet_kelly_naive(self):
        opt = MarkowitzOptimizer()
        bets = [
            BetProposal(
                bet_id="b0",
                description="Solo",
                odds=2.5,
                model_prob=0.55,
                correlation_group="grp",
            )
        ]
        allocation = opt.kelly_naive(bets)
        assert len(allocation.weights) == 1
        assert allocation.weights[0] >= 0.0


# ---------------------------------------------------------------------------
# MarkowitzOptimizer — optimise (Sharpe maximisation)
# ---------------------------------------------------------------------------


class TestMarkowitzOptimizerOptimise:

    def test_optimise_returns_allocation(self):
        opt = MarkowitzOptimizer()
        allocation = opt.optimise(_make_bets(3))
        assert allocation is not None

    def test_optimise_weights_nonnegative(self):
        opt = MarkowitzOptimizer()
        allocation = opt.optimise(_make_bets(3))
        for w in allocation.weights:
            assert w >= -1e-9  # allow tiny floating-point negatives

    def test_optimise_excludes_negative_ev_bets(self):
        # All-negative EV bets should yield zero weights
        opt = MarkowitzOptimizer()
        bets = [
            BetProposal(
                bet_id="b0",
                description="Bad bet",
                odds=1.5,
                model_prob=0.30,
                correlation_group="grp",
            ),
        ]
        allocation = opt.optimise(bets)
        assert all(w == pytest.approx(0.0, abs=1e-9) for w in allocation.weights)

    def test_psd_regularisation_does_not_crash(self):
        # Same-match bets share correlation → potential non-PSD covariance
        opt = MarkowitzOptimizer()
        bets = []
        for i in range(5):
            bets.append(
                BetProposal(
                    bet_id=f"b{i}",
                    description=f"Match bet {i}",
                    odds=2.1,
                    model_prob=0.55,
                    correlation_group="same_group",
                    same_match_group="match_x",
                )
            )
        # Should not raise even with highly correlated bets
        allocation = opt.optimise(bets)
        assert allocation is not None


# ---------------------------------------------------------------------------
# optimise_portfolio convenience function
# ---------------------------------------------------------------------------


class TestOptimisePortfolio:

    def test_returns_allocation_object(self):
        bets_dicts = [
            {
                "bet_id": "b1",
                "description": "Napoli Win",
                "odds": 2.10,
                "model_prob": 0.54,
                "correlation_group": "serie_a_r28",
            },
            {
                "bet_id": "b2",
                "description": "Lazio Win",
                "odds": 3.20,
                "model_prob": 0.38,
                "correlation_group": "serie_a_r28",
            },
        ]
        allocation = optimise_portfolio(bets_dicts, bankroll=1000.0)
        assert allocation is not None

    def test_weights_nonnegative(self):
        bets_dicts = [
            {
                "bet_id": "b1",
                "description": "Test",
                "odds": 2.5,
                "model_prob": 0.55,
                "correlation_group": "grp",
            },
        ]
        allocation = optimise_portfolio(bets_dicts, bankroll=500.0)
        for w in allocation.weights:
            assert w >= -1e-9

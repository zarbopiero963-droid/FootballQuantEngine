import pandas as pd

from engine.confidence_engine import ConfidenceEngine
from engine.market_comparison import MarketComparison
from engine.model_agreement import ModelAgreement
from engine.probability_calibrator import ProbabilityCalibrator
from strategies.no_bet_filter import NoBetFilter
from training.backtest_metrics import BacktestMetrics


def test_probability_calibrator_normalizes():
    calibrator = ProbabilityCalibrator()

    result = calibrator.calibrate(
        {
            "home_win": 0.51,
            "draw": 0.24,
            "away_win": 0.25,
        }
    )

    total = result["home_win"] + result["draw"] + result["away_win"]

    assert 0.999 <= total <= 1.001


def test_model_agreement_range():
    score = ModelAgreement().calculate(
        [
            {"home_win": 0.50, "draw": 0.25, "away_win": 0.25},
            {"home_win": 0.52, "draw": 0.24, "away_win": 0.24},
            {"home_win": 0.48, "draw": 0.27, "away_win": 0.25},
        ]
    )

    assert 0.0 <= score <= 1.0


def test_market_implied_probabilities_normalized():
    market = MarketComparison()

    implied = market.implied_probabilities({"home": 2.0, "draw": 3.2, "away": 3.8})

    total = implied["home_win"] + implied["draw"] + implied["away_win"]

    assert 0.999 <= total <= 1.001


def test_market_edge_map_contains_expected_keys():
    market = MarketComparison()

    edge_map = market.edge_vs_market(
        {
            "home_win": 0.55,
            "draw": 0.25,
            "away_win": 0.20,
        },
        {
            "home_win": 0.50,
            "draw": 0.28,
            "away_win": 0.22,
        },
    )

    assert "home_win" in edge_map
    assert "draw" in edge_map
    assert "away_win" in edge_map


def test_confidence_engine_range():
    score = ConfidenceEngine().score(
        probability=0.58,
        edge=0.07,
        agreement=0.70,
    )

    assert 0.0 <= score <= 1.0


def test_no_bet_filter_outputs_valid_state():
    decision = NoBetFilter().decide(
        probability=0.57,
        edge=0.06,
        confidence=0.72,
        agreement=0.68,
    )

    assert decision in ("BET", "WATCHLIST", "NO_BET")


def test_backtest_metrics_real_values():
    df = pd.DataFrame(
        [
            {
                "home_prob": 0.60,
                "draw_prob": 0.20,
                "away_prob": 0.20,
                "actual_home_win": 1,
                "actual_draw": 0,
                "actual_away_win": 0,
                "home_odds": 2.10,
                "draw_odds": 3.20,
                "away_odds": 3.80,
                "home_goals": 2,
                "away_goals": 1,
            },
            {
                "home_prob": 0.40,
                "draw_prob": 0.35,
                "away_prob": 0.25,
                "actual_home_win": 0,
                "actual_draw": 1,
                "actual_away_win": 0,
                "home_odds": 2.40,
                "draw_odds": 3.10,
                "away_odds": 2.90,
                "home_goals": 1,
                "away_goals": 1,
            },
        ]
    )

    metrics = BacktestMetrics().calculate(df)

    assert "roi" in metrics
    assert "yield" in metrics
    assert "hit_rate" in metrics
    assert "max_drawdown" in metrics
    assert "bankroll_history" in metrics
    assert "accuracy_history" in metrics
    assert metrics["total_bets"] == 2

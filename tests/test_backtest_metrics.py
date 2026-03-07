import pandas as pd

from training.backtest_metrics import BacktestMetrics


def test_backtest_metrics_runs():

    df = pd.DataFrame(
        [
            {
                "home_prob": 0.60,
                "draw_prob": 0.20,
                "away_prob": 0.20,
                "home_goals": 2,
                "away_goals": 1,
                "actual_home_win": 1,
                "actual_draw": 0,
                "actual_away_win": 0,
                "home_odds": 2.10,
                "draw_odds": 3.20,
                "away_odds": 3.80,
            }
        ]
    )

    metrics = BacktestMetrics().calculate(df)

    assert "roi" in metrics
    assert "yield" in metrics
    assert "hit_rate" in metrics
    assert "max_drawdown" in metrics

import pandas as pd

from analytics.market_inefficiency_scanner import MarketInefficiencyScanner


def test_market_inefficiency_runs():

    df = pd.DataFrame(
        [
            {
                "team": "TeamA",
                "home_win": 0.60,
                "draw": 0.20,
                "away_win": 0.20,
            },
            {
                "team": "TeamB",
                "home_win": 0.40,
                "draw": 0.30,
                "away_win": 0.30,
            },
        ]
    )

    result = MarketInefficiencyScanner().scan(df)

    assert isinstance(result, pd.DataFrame)
    assert "inefficiency_score" in result.columns

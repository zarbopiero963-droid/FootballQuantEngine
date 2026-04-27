import pandas as pd

from analytics.market_pressure import MarketPressureAnalyzer


def test_market_pressure_runs():

    df = pd.DataFrame(
        [
            {
                "fixture_id": 1,
                "timestamp": "2026-03-06T10:00:00",
                "home_odds": 2.1,
                "draw_odds": 3.2,
                "away_odds": 3.5,
            },
            {
                "fixture_id": 1,
                "timestamp": "2026-03-06T12:00:00",
                "home_odds": 2.0,
                "draw_odds": 3.3,
                "away_odds": 3.6,
            },
        ]
    )

    analyzer = MarketPressureAnalyzer()

    result = analyzer.calculate(df)

    assert isinstance(result, pd.DataFrame)
    assert "home_pressure" in result.columns

import pandas as pd

from engine.offline_engine import OfflineEngine


def test_offline_engine_analysis_runs():

    fixtures_df = pd.DataFrame(
        [
            {
                "home": "TeamA",
                "away": "TeamB",
                "home_goals": 2,
                "away_goals": 1,
            },
            {
                "home": "TeamB",
                "away": "TeamA",
                "home_goals": 0,
                "away_goals": 1,
            },
        ]
    )

    engine = OfflineEngine()

    result = engine.run_offline_analysis(fixtures_df)

    assert "features" in result
    assert "predictions" in result
    assert "market_inefficiency" in result

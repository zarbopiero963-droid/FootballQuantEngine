import pandas as pd

from engine.prediction_pipeline import PredictionPipeline


def test_prediction_pipeline_runs():

    features_df = pd.DataFrame(
        [
            {
                "home": "TeamA",
                "away": "TeamB",
                "home_attack": 1.4,
                "home_defense": 1.0,
                "away_attack": 1.2,
                "away_defense": 1.1,
                "elo_diff": 0,
            }
        ]
    )

    odds_data = {"TeamA_vs_TeamB": {"home": 2.0, "draw": 3.2, "away": 3.8}}

    pipeline = PredictionPipeline()

    value_bets = pipeline.run(features_df, odds_data)

    assert isinstance(value_bets, list)

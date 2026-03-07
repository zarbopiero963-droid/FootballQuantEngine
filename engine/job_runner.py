import pandas as pd

from engine.prediction_pipeline import PredictionPipeline


class JobRunner:

    def __init__(self):

        self.pipeline = PredictionPipeline()

    def _sample_features(self):

        return pd.DataFrame(
            [
                {
                    "home": "TeamA",
                    "away": "TeamB",
                    "home_attack": 1.45,
                    "home_defense": 0.92,
                    "away_attack": 1.12,
                    "away_defense": 1.08,
                    "elo_diff": 55,
                },
                {
                    "home": "TeamC",
                    "away": "TeamD",
                    "home_attack": 1.10,
                    "home_defense": 1.04,
                    "away_attack": 1.05,
                    "away_defense": 1.01,
                    "elo_diff": 8,
                },
                {
                    "home": "TeamE",
                    "away": "TeamF",
                    "home_attack": 1.62,
                    "home_defense": 0.88,
                    "away_attack": 0.94,
                    "away_defense": 1.16,
                    "elo_diff": 102,
                },
            ]
        )

    def _sample_odds(self):

        return {
            "TeamA_vs_TeamB": {
                "home": 2.05,
                "draw": 3.35,
                "away": 3.90,
            },
            "TeamC_vs_TeamD": {
                "home": 2.60,
                "draw": 3.00,
                "away": 2.85,
            },
            "TeamE_vs_TeamF": {
                "home": 1.82,
                "draw": 3.55,
                "away": 4.70,
            },
        }

    def run_cycle(self):

        features_df = self._sample_features()
        odds_data = self._sample_odds()

        results = self.pipeline.run(
            features_df=features_df,
            odds_data=odds_data,
        )

        return results

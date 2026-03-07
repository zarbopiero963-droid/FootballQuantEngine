from __future__ import annotations

import pandas as pd

from training.backtest_engine import BacktestEngine


class OfflineEngine:

    def __init__(self):
        self.backtest_engine = BacktestEngine()

    def analyze(self, date_from=None, date_to=None, league=None):
        return self.backtest_engine.run(
            date_from=date_from,
            date_to=date_to,
            league=league,
        )

    def run_offline_analysis(self, fixtures_df: pd.DataFrame):

        if fixtures_df is None or fixtures_df.empty:
            return {
                "features": pd.DataFrame(),
                "predictions": pd.DataFrame(),
                "market_inefficiency": pd.DataFrame(),
                "rows": 0,
            }

        df = fixtures_df.copy()

        if "home" in df.columns and "home_team" not in df.columns:
            df["home_team"] = df["home"]

        if "away" in df.columns and "away_team" not in df.columns:
            df["away_team"] = df["away"]

        if "result" not in df.columns:
            df["result"] = (df["home_goals"] > df["away_goals"]).map(
                {True: "H", False: "A"}
            )
            df.loc[df["home_goals"] == df["away_goals"], "result"] = "D"

        features = df.copy()
        features["goal_diff"] = features["home_goals"] - features["away_goals"]
        features["total_goals"] = features["home_goals"] + features["away_goals"]

        predictions = pd.DataFrame(
            {
                "home_team": features["home_team"],
                "away_team": features["away_team"],
                "predicted_result": features["result"],
            }
        )

        market_inefficiency = pd.DataFrame(
            {
                "home_team": features["home_team"],
                "away_team": features["away_team"],
                "inefficiency_score": 0.0,
            }
        )

        return {
            "features": features,
            "predictions": predictions,
            "market_inefficiency": market_inefficiency,
            "rows": len(features.index),
        }

    def run_reports(self):
        return True

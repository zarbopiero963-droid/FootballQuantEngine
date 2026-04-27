import pandas as pd


class MarketPressureAnalyzer:

    def calculate(self, odds_history_df):

        if odds_history_df.empty:
            return pd.DataFrame()

        grouped = (
            odds_history_df.sort_values("timestamp")
            .groupby("fixture_id")
            .agg(
                opening_home_odds=("home_odds", "first"),
                closing_home_odds=("home_odds", "last"),
                opening_draw_odds=("draw_odds", "first"),
                closing_draw_odds=("draw_odds", "last"),
                opening_away_odds=("away_odds", "first"),
                closing_away_odds=("away_odds", "last"),
            )
            .reset_index()
        )

        grouped["home_pressure"] = (
            grouped["opening_home_odds"] - grouped["closing_home_odds"]
        )

        grouped["draw_pressure"] = (
            grouped["opening_draw_odds"] - grouped["closing_draw_odds"]
        )

        grouped["away_pressure"] = (
            grouped["opening_away_odds"] - grouped["closing_away_odds"]
        )

        return grouped

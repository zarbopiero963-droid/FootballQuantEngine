import pandas as pd


class LeaguePredictability:

    def score(self, fixtures_df):

        if fixtures_df.empty:
            return pd.DataFrame(columns=["league", "predictability_score"])

        rows = []

        for league in fixtures_df["league"].dropna().unique():

            league_df = fixtures_df[fixtures_df["league"] == league].copy()

            if league_df.empty:
                continue

            home_wins = (league_df["home_goals"] > league_df["away_goals"]).mean()

            draws = (league_df["home_goals"] == league_df["away_goals"]).mean()

            away_wins = (league_df["home_goals"] < league_df["away_goals"]).mean()

            score = max(home_wins, draws, away_wins)

            rows.append(
                {
                    "league": league,
                    "predictability_score": score,
                }
            )

        result = pd.DataFrame(rows)

        if result.empty:
            return result

        return result.sort_values(
            by="predictability_score",
            ascending=False,
        )

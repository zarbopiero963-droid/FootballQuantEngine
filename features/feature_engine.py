import pandas as pd

from features.elo_features import elo_diff
from features.team_stats import build_team_stats


class FeatureEngine:

    def __init__(self, elo_df=None):

        self.elo_df = elo_df if elo_df is not None else pd.DataFrame()

    def build_features(self, fixtures_df):

        team_stats = build_team_stats(fixtures_df)

        rows = []

        for _, row in fixtures_df.iterrows():

            home = row["home"]
            away = row["away"]

            home_stats = team_stats[team_stats["team"] == home]
            away_stats = team_stats[team_stats["team"] == away]

            if home_stats.empty or away_stats.empty:
                continue

            home_attack = home_stats.iloc[0]["goals_for"]
            home_defense = home_stats.iloc[0]["goals_against"]

            away_attack = away_stats.iloc[0]["goals_for"]
            away_defense = away_stats.iloc[0]["goals_against"]

            elo_difference = elo_diff(home, away, self.elo_df)

            rows.append(
                {
                    "home": home,
                    "away": away,
                    "home_attack": home_attack,
                    "home_defense": home_defense,
                    "away_attack": away_attack,
                    "away_defense": away_defense,
                    "elo_diff": elo_difference,
                }
            )

        return pd.DataFrame(rows)

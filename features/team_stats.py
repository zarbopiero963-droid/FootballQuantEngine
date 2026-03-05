import pandas as pd


def build_team_stats(fixtures_df):

    if fixtures_df.empty:
        return pd.DataFrame()

    df = fixtures_df.copy()

    home = df[["home", "home_goals", "away_goals"]].copy()
    home.columns = ["team", "goals_for", "goals_against"]

    away = df[["away", "away_goals", "home_goals"]].copy()
    away.columns = ["team", "goals_for", "goals_against"]

    teams = pd.concat([home, away])

    stats = (
        teams.groupby("team")
        .agg(
            goals_for=("goals_for", "mean"),
            goals_against=("goals_against", "mean"),
            matches=("goals_for", "count"),
        )
        .reset_index()
    )

    return stats

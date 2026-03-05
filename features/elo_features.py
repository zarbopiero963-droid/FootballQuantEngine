

def elo_diff(home_team, away_team, elo_df):

    if elo_df.empty:
        return 0

    home_row = elo_df[elo_df["Club"] == home_team]
    away_row = elo_df[elo_df["Club"] == away_team]

    if home_row.empty or away_row.empty:
        return 0

    home_elo = home_row.iloc[0]["Elo"]
    away_elo = away_row.iloc[0]["Elo"]

    return home_elo - away_elo

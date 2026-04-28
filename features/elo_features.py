from __future__ import annotations

import pandas as pd


def elo_diff(home_team: str, away_team: str, elo_df: pd.DataFrame) -> float:
    if elo_df.empty:
        return 0.0

    home_row = elo_df[elo_df["Club"] == home_team]
    away_row = elo_df[elo_df["Club"] == away_team]

    if home_row.empty or away_row.empty:
        return 0.0

    home_elo: float = float(home_row.iloc[0]["Elo"])
    away_elo: float = float(away_row.iloc[0]["Elo"])

    return home_elo - away_elo

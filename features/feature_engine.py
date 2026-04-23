"""
Feature engine: builds a normalised feature matrix from fixture data.

Now includes:
  - League-strength normalization (attack/defense rates on universal scale)
  - ELO-based rating difference
  - Rolling team statistics (goals, win rates)
  - xG features (when available)
"""

from __future__ import annotations

import pandas as pd

from features.elo_features import elo_diff
from features.team_stats import build_team_stats
from normalization.league_strength import LeagueStrengthNormalizer


class FeatureEngine:
    """
    Builds a normalised feature DataFrame from a fixtures DataFrame.

    Parameters
    ----------
    elo_df      : optional ELO ratings DataFrame (columns: Club, Elo)
    normalizer  : optional LeagueStrengthNormalizer instance;
                  a default one is created if not supplied.
    """

    def __init__(
        self,
        elo_df=None,
        normalizer: LeagueStrengthNormalizer | None = None,
    ) -> None:
        self.elo_df = elo_df if elo_df is not None else pd.DataFrame()
        self.normalizer = normalizer or LeagueStrengthNormalizer()

    def build_features(
        self,
        fixtures_df: pd.DataFrame,
        league_col: str = "league",
    ) -> pd.DataFrame:
        """
        Build the feature matrix from a fixtures DataFrame.

        Parameters
        ----------
        fixtures_df : pd.DataFrame
            Must contain columns: home, away, home_goals, away_goals.
            Optional: league (for strength normalisation), xg_home, xg_away.
        league_col  : name of the league column.

        Returns
        -------
        pd.DataFrame with one row per fixture, containing:
          home, away, league,
          home_attack, home_defense, away_attack, away_defense,  (normalised)
          home_attack_raw, away_attack_raw,                      (league-local)
          elo_diff,
          home_xg, away_xg,                                      (if available)
          league_coefficient.
        """
        if fixtures_df.empty:
            return pd.DataFrame()

        team_stats = build_team_stats(fixtures_df)

        rows = []

        for _, row in fixtures_df.iterrows():
            home = row["home"]
            away = row["away"]
            league = str(row.get(league_col, "Unknown") or "Unknown")

            home_stats = team_stats[team_stats["team"] == home]
            away_stats = team_stats[team_stats["team"] == away]

            if home_stats.empty or away_stats.empty:
                continue

            # Raw (league-local) attack and defense rates
            home_att_raw = float(home_stats.iloc[0]["goals_for"])
            home_def_raw = float(home_stats.iloc[0]["goals_against"])
            away_att_raw = float(away_stats.iloc[0]["goals_for"])
            away_def_raw = float(away_stats.iloc[0]["goals_against"])

            # League-normalised rates
            coeff = self.normalizer.coefficient(league)
            home_att_norm = self.normalizer.normalize_attack(home_att_raw, league)
            home_def_norm = self.normalizer.normalize_defense(home_def_raw, league)
            away_att_norm = self.normalizer.normalize_attack(away_att_raw, league)
            away_def_norm = self.normalizer.normalize_defense(away_def_raw, league)

            elo_difference = elo_diff(home, away, self.elo_df)

            # xG features (optional)
            home_xg = float(row.get("xg_home") or row.get("xg") or home_att_raw)
            away_xg = float(row.get("xg_away") or away_att_raw)

            rows.append(
                {
                    "home": home,
                    "away": away,
                    "league": league,
                    "home_attack": home_att_norm,
                    "home_defense": home_def_norm,
                    "away_attack": away_att_norm,
                    "away_defense": away_def_norm,
                    "home_attack_raw": home_att_raw,
                    "away_attack_raw": away_att_raw,
                    "elo_diff": elo_difference,
                    "home_xg": home_xg,
                    "away_xg": away_xg,
                    "league_coefficient": coeff,
                }
            )

        return pd.DataFrame(rows)

from __future__ import annotations

import pandas as pd


class OfflineFeatureBuilder:

    def build_team_offline_features(self, fixtures_df):

        if fixtures_df.empty:
            return pd.DataFrame()

        rows = []

        teams = set(fixtures_df["home"]).union(set(fixtures_df["away"]))

        for team in teams:

            home_df = fixtures_df[fixtures_df["home"] == team].copy()
            away_df = fixtures_df[fixtures_df["away"] == team].copy()

            home_played = len(home_df)
            away_played = len(away_df)
            total_played = home_played + away_played

            if total_played == 0:
                continue

            goals_for = home_df["home_goals"].sum() + away_df["away_goals"].sum()
            goals_against = home_df["away_goals"].sum() + away_df["home_goals"].sum()

            clean_sheets = (home_df["away_goals"] == 0).sum() + (
                away_df["home_goals"] == 0
            ).sum()

            over_25 = ((home_df["home_goals"] + home_df["away_goals"]) > 2).sum() + (
                (away_df["home_goals"] + away_df["away_goals"]) > 2
            ).sum()

            btts = ((home_df["home_goals"] > 0) & (home_df["away_goals"] > 0)).sum() + (
                (away_df["home_goals"] > 0) & (away_df["away_goals"] > 0)
            ).sum()

            recent_home = home_df.tail(5)
            recent_away = away_df.tail(5)

            recent_df = pd.concat([recent_home, recent_away]).tail(10)

            rolling_goals_for = (
                recent_home["home_goals"].tolist() + recent_away["away_goals"].tolist()
            )

            rolling_goals_against = (
                recent_home["away_goals"].tolist() + recent_away["home_goals"].tolist()
            )

            weighted_form_score = 0.0

            recent_matches = recent_df.to_dict("records")

            weight = 1.0

            for match in reversed(recent_matches):

                if match["home"] == team:
                    gf = match["home_goals"]
                    ga = match["away_goals"]
                else:
                    gf = match["away_goals"]
                    ga = match["home_goals"]

                if gf > ga:
                    result_points = 3
                elif gf == ga:
                    result_points = 1
                else:
                    result_points = 0

                weighted_form_score += result_points * weight
                weight *= 0.85

            rows.append(
                {
                    "team": team,
                    "matches_played": total_played,
                    "goals_for_avg": goals_for / total_played,
                    "goals_against_avg": goals_against / total_played,
                    "clean_sheet_rate": clean_sheets / total_played,
                    "over_2_5_rate": over_25 / total_played,
                    "btts_rate": btts / total_played,
                    "rolling_goals_for_avg": (
                        sum(rolling_goals_for) / len(rolling_goals_for)
                        if rolling_goals_for
                        else 0.0
                    ),
                    "rolling_goals_against_avg": (
                        sum(rolling_goals_against) / len(rolling_goals_against)
                        if rolling_goals_against
                        else 0.0
                    ),
                    "weighted_form_score": weighted_form_score,
                }
            )

        return pd.DataFrame(rows)

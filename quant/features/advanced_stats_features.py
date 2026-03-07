class AdvancedStatsFeatureBuilder:

    def __init__(self, stats_by_team=None):
        self.stats_by_team = stats_by_team or {}

    def _team_stats(self, team_name):
        return self.stats_by_team.get(
            team_name,
            {
                "xg_for": 1.2,
                "xg_against": 1.2,
                "xpts": 1.4,
                "shots": 10.0,
                "shots_on_target": 3.5,
            },
        )

    def build(self, home_team, away_team):
        home = self._team_stats(home_team)
        away = self._team_stats(away_team)

        return {
            "home_xg_for": float(home.get("xg_for", 1.2)),
            "away_xg_for": float(away.get("xg_for", 1.2)),
            "home_xg_against": float(home.get("xg_against", 1.2)),
            "away_xg_against": float(away.get("xg_against", 1.2)),
            "xg_diff": float(home.get("xg_for", 1.2)) - float(away.get("xg_for", 1.2)),
            "xga_diff": float(away.get("xg_against", 1.2))
            - float(home.get("xg_against", 1.2)),
            "xpts_diff": float(home.get("xpts", 1.4)) - float(away.get("xpts", 1.4)),
            "shots_diff": float(home.get("shots", 10.0))
            - float(away.get("shots", 10.0)),
            "sot_diff": float(home.get("shots_on_target", 3.5))
            - float(away.get("shots_on_target", 3.5)),
        }

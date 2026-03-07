class SampleAPIFootballClient:

    def get_completed_matches(self, league=None, season=None):
        return [
            {
                "fixture_id": "m1",
                "home_team": "Inter",
                "away_team": "Milan",
                "home_goals": 2,
                "away_goals": 1,
            },
            {
                "fixture_id": "m2",
                "home_team": "Juventus",
                "away_team": "Roma",
                "home_goals": 1,
                "away_goals": 1,
            },
            {
                "fixture_id": "m3",
                "home_team": "Napoli",
                "away_team": "Lazio",
                "home_goals": 3,
                "away_goals": 0,
            },
            {
                "fixture_id": "m4",
                "home_team": "Milan",
                "away_team": "Juventus",
                "home_goals": 0,
                "away_goals": 1,
            },
            {
                "fixture_id": "m5",
                "home_team": "Roma",
                "away_team": "Inter",
                "home_goals": 1,
                "away_goals": 2,
            },
            {
                "fixture_id": "m6",
                "home_team": "Lazio",
                "away_team": "Napoli",
                "home_goals": 1,
                "away_goals": 2,
            },
            {
                "fixture_id": "m7",
                "home_team": "Inter",
                "away_team": "Juventus",
                "home_goals": 2,
                "away_goals": 0,
            },
            {
                "fixture_id": "m8",
                "home_team": "Milan",
                "away_team": "Roma",
                "home_goals": 2,
                "away_goals": 2,
            },
            {
                "fixture_id": "m9",
                "home_team": "Napoli",
                "away_team": "Inter",
                "home_goals": 1,
                "away_goals": 1,
            },
            {
                "fixture_id": "m10",
                "home_team": "Lazio",
                "away_team": "Milan",
                "home_goals": 0,
                "away_goals": 1,
            },
        ]

    def get_upcoming_matches(self, league=None, season=None):
        return [
            {
                "fixture_id": "u1",
                "home_team": "Inter",
                "away_team": "Roma",
            },
            {
                "fixture_id": "u2",
                "home_team": "Napoli",
                "away_team": "Milan",
            },
            {
                "fixture_id": "u3",
                "home_team": "Juventus",
                "away_team": "Lazio",
            },
        ]

    def get_prematch_odds(self, fixture_ids):
        return {
            "u1": {"home": 1.88, "draw": 3.55, "away": 4.40},
            "u2": {"home": 2.25, "draw": 3.25, "away": 3.05},
            "u3": {"home": 2.05, "draw": 3.15, "away": 3.75},
        }


class SampleUnderstatClient:

    def get_team_advanced_stats(self, league=None, season=None):
        return {
            "Inter": {
                "xg_for": 1.85,
                "xg_against": 0.90,
                "xpts": 2.10,
                "shots": 14.0,
                "shots_on_target": 5.3,
            },
            "Roma": {
                "xg_for": 1.30,
                "xg_against": 1.20,
                "xpts": 1.50,
                "shots": 11.0,
                "shots_on_target": 4.1,
            },
            "Napoli": {
                "xg_for": 1.70,
                "xg_against": 1.00,
                "xpts": 1.95,
                "shots": 13.2,
                "shots_on_target": 4.8,
            },
            "Milan": {
                "xg_for": 1.48,
                "xg_against": 1.08,
                "xpts": 1.72,
                "shots": 12.0,
                "shots_on_target": 4.4,
            },
            "Juventus": {
                "xg_for": 1.42,
                "xg_against": 0.94,
                "xpts": 1.80,
                "shots": 11.5,
                "shots_on_target": 4.0,
            },
            "Lazio": {
                "xg_for": 1.18,
                "xg_against": 1.16,
                "xpts": 1.35,
                "shots": 10.4,
                "shots_on_target": 3.7,
            },
        }

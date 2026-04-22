class SampleAPIFootballClient:

    def get_completed_matches(self, league=None, season=None):
        return [
            {
                "fixture_id": "m1",
                "home_team": "Inter",
                "home_team_id": 505,
                "away_team": "Milan",
                "away_team_id": 489,
                "home_goals": 2,
                "away_goals": 1,
                "match_date": "2024-09-15T20:45:00+00:00",
                "referee": "Daniele Orsato",
            },
            {
                "fixture_id": "m2",
                "home_team": "Juventus",
                "home_team_id": 496,
                "away_team": "Roma",
                "away_team_id": 497,
                "home_goals": 1,
                "away_goals": 1,
                "match_date": "2024-09-22T18:00:00+00:00",
                "referee": "Massimiliano Irrati",
            },
            {
                "fixture_id": "m3",
                "home_team": "Napoli",
                "home_team_id": 492,
                "away_team": "Lazio",
                "away_team_id": 487,
                "home_goals": 3,
                "away_goals": 0,
                "match_date": "2024-09-29T20:45:00+00:00",
                "referee": "Daniele Orsato",
            },
            {
                "fixture_id": "m4",
                "home_team": "Milan",
                "home_team_id": 489,
                "away_team": "Juventus",
                "away_team_id": 496,
                "home_goals": 0,
                "away_goals": 1,
                "match_date": "2024-10-06T20:45:00+00:00",
                "referee": "Massimiliano Irrati",
            },
            {
                "fixture_id": "m5",
                "home_team": "Roma",
                "home_team_id": 497,
                "away_team": "Inter",
                "away_team_id": 505,
                "home_goals": 1,
                "away_goals": 2,
                "match_date": "2024-10-13T20:45:00+00:00",
                "referee": "Daniele Orsato",
            },
            {
                "fixture_id": "m6",
                "home_team": "Lazio",
                "home_team_id": 487,
                "away_team": "Napoli",
                "away_team_id": 492,
                "home_goals": 1,
                "away_goals": 2,
                "match_date": "2024-10-20T15:00:00+00:00",
                "referee": "Maurizio Mariani",
            },
            {
                "fixture_id": "m7",
                "home_team": "Inter",
                "home_team_id": 505,
                "away_team": "Juventus",
                "away_team_id": 496,
                "home_goals": 2,
                "away_goals": 0,
                "match_date": "2024-10-27T20:45:00+00:00",
                "referee": "Maurizio Mariani",
            },
            {
                "fixture_id": "m8",
                "home_team": "Milan",
                "home_team_id": 489,
                "away_team": "Roma",
                "away_team_id": 497,
                "home_goals": 2,
                "away_goals": 2,
                "match_date": "2024-11-03T20:45:00+00:00",
                "referee": "Daniele Orsato",
            },
            {
                "fixture_id": "m9",
                "home_team": "Napoli",
                "home_team_id": 492,
                "away_team": "Inter",
                "away_team_id": 505,
                "home_goals": 1,
                "away_goals": 1,
                "match_date": "2024-11-10T20:45:00+00:00",
                "referee": "Massimiliano Irrati",
            },
            {
                "fixture_id": "m10",
                "home_team": "Lazio",
                "home_team_id": 487,
                "away_team": "Milan",
                "away_team_id": 489,
                "home_goals": 0,
                "away_goals": 1,
                "match_date": "2024-11-17T15:00:00+00:00",
                "referee": "Maurizio Mariani",
            },
        ]

    def get_upcoming_matches(self, league=None, season=None):
        return [
            {
                "fixture_id": "u1",
                "home_team": "Inter",
                "home_team_id": 505,
                "away_team": "Roma",
                "away_team_id": 497,
                "match_date": "2024-12-01T20:45:00+00:00",
                "referee": "Daniele Orsato",
            },
            {
                "fixture_id": "u2",
                "home_team": "Napoli",
                "home_team_id": 492,
                "away_team": "Milan",
                "away_team_id": 489,
                "match_date": "2024-12-01T18:00:00+00:00",
                "referee": "Massimiliano Irrati",
            },
            {
                "fixture_id": "u3",
                "home_team": "Juventus",
                "home_team_id": 496,
                "away_team": "Lazio",
                "away_team_id": 487,
                "match_date": "2024-12-01T15:00:00+00:00",
                "referee": "Maurizio Mariani",
            },
        ]

    def get_prematch_odds(self, fixture_ids):
        return {
            "u1": {"home": 1.88, "draw": 3.55, "away": 4.40},
            "u2": {"home": 2.25, "draw": 3.25, "away": 3.05},
            "u3": {"home": 2.05, "draw": 3.15, "away": 3.75},
        }

    def get_standings(self, league=None, season=None):
        return [
            {"team": "Inter", "position": 1, "points": 35, "goal_diff": 20},
            {"team": "Napoli", "position": 2, "points": 30, "goal_diff": 14},
            {"team": "Juventus", "position": 3, "points": 27, "goal_diff": 9},
            {"team": "Milan", "position": 4, "points": 25, "goal_diff": 5},
            {"team": "Roma", "position": 8, "points": 18, "goal_diff": 1},
            {"team": "Lazio", "position": 9, "points": 17, "goal_diff": -2},
        ]

    def get_injuries(self, league=None, season=None):
        return {
            "Milan": [
                {"player": "Mike Maignan", "position": "Goalkeeper", "type": "Muscle"},
            ],
            "Roma": [
                {"player": "Paulo Dybala", "position": "Attacker", "type": "Muscle"},
                {"player": "Romelu Lukaku", "position": "Attacker", "type": "Knee"},
            ],
        }

    def get_referee_stats(self, league=None, season=None):
        return {
            "Daniele Orsato": {"home_bias": 0.05, "strictness": 1.1, "games": 20},
            "Massimiliano Irrati": {"home_bias": -0.03, "strictness": 0.9, "games": 15},
            "Maurizio Mariani": {"home_bias": 0.01, "strictness": 1.0, "games": 18},
        }

    def get_team_xg_averages(self, league=None, season=None):
        return {
            "Inter": {"xg_for": 1.85, "xg_against": 0.90, "shots": 14.0, "shots_on_target": 5.3},
            "Roma": {"xg_for": 1.30, "xg_against": 1.20, "shots": 11.0, "shots_on_target": 4.1},
            "Napoli": {"xg_for": 1.70, "xg_against": 1.00, "shots": 13.2, "shots_on_target": 4.8},
            "Milan": {"xg_for": 1.48, "xg_against": 1.08, "shots": 12.0, "shots_on_target": 4.4},
            "Juventus": {"xg_for": 1.42, "xg_against": 0.94, "shots": 11.5, "shots_on_target": 4.0},
            "Lazio": {"xg_for": 1.18, "xg_against": 1.16, "shots": 10.4, "shots_on_target": 3.7},
        }

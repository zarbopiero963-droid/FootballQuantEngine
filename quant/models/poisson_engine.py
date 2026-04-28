from __future__ import annotations

import math


class PoissonEngine:

    def __init__(self, max_goals=8):
        self.max_goals = max_goals
        self.league_avg_home_goals = 1.35
        self.league_avg_away_goals = 1.10
        self.team_stats = {}

    def _poisson_pmf(self, goals, lam):
        return (math.exp(-lam) * (lam**goals)) / math.factorial(goals)

    def fit(self, completed_matches):
        if not completed_matches:
            self.team_stats = {}
            return

        total_home_goals = 0.0
        total_away_goals = 0.0
        match_count = 0

        raw = {}

        for match in completed_matches:
            home = match["home_team"]
            away = match["away_team"]
            hg = float(match["home_goals"])
            ag = float(match["away_goals"])

            total_home_goals += hg
            total_away_goals += ag
            match_count += 1

            if home not in raw:
                raw[home] = {
                    "home_scored": 0.0,
                    "home_conceded": 0.0,
                    "home_matches": 0,
                    "away_scored": 0.0,
                    "away_conceded": 0.0,
                    "away_matches": 0,
                }

            if away not in raw:
                raw[away] = {
                    "home_scored": 0.0,
                    "home_conceded": 0.0,
                    "home_matches": 0,
                    "away_scored": 0.0,
                    "away_conceded": 0.0,
                    "away_matches": 0,
                }

            raw[home]["home_scored"] += hg
            raw[home]["home_conceded"] += ag
            raw[home]["home_matches"] += 1

            raw[away]["away_scored"] += ag
            raw[away]["away_conceded"] += hg
            raw[away]["away_matches"] += 1

        self.league_avg_home_goals = (
            total_home_goals / match_count if match_count else 1.35
        )
        self.league_avg_away_goals = (
            total_away_goals / match_count if match_count else 1.10
        )

        fitted = {}

        for team, values in raw.items():
            home_matches = max(values["home_matches"], 1)
            away_matches = max(values["away_matches"], 1)

            avg_home_scored = values["home_scored"] / home_matches
            avg_home_conceded = values["home_conceded"] / home_matches
            avg_away_scored = values["away_scored"] / away_matches
            avg_away_conceded = values["away_conceded"] / away_matches

            fitted[team] = {
                "attack_home": (
                    avg_home_scored / self.league_avg_home_goals
                    if self.league_avg_home_goals
                    else 1.0
                ),
                "defense_home": (
                    avg_home_conceded / self.league_avg_away_goals
                    if self.league_avg_away_goals
                    else 1.0
                ),
                "attack_away": (
                    avg_away_scored / self.league_avg_away_goals
                    if self.league_avg_away_goals
                    else 1.0
                ),
                "defense_away": (
                    avg_away_conceded / self.league_avg_home_goals
                    if self.league_avg_home_goals
                    else 1.0
                ),
            }

        self.team_stats = fitted

    def get_team_profile(self, team_name):
        return self.team_stats.get(
            team_name,
            {
                "attack_home": 1.0,
                "defense_home": 1.0,
                "attack_away": 1.0,
                "defense_away": 1.0,
            },
        )

    def expected_goals(self, home_team, away_team):
        home_profile = self.get_team_profile(home_team)
        away_profile = self.get_team_profile(away_team)

        lambda_home = (
            self.league_avg_home_goals
            * home_profile["attack_home"]
            * away_profile["defense_away"]
        )
        lambda_away = (
            self.league_avg_away_goals
            * away_profile["attack_away"]
            * home_profile["defense_home"]
        )

        lambda_home = max(0.2, lambda_home)
        lambda_away = max(0.2, lambda_away)

        return lambda_home, lambda_away

    def score_matrix(self, home_team, away_team):
        lambda_home, lambda_away = self.expected_goals(home_team, away_team)

        matrix = []
        for hg in range(self.max_goals + 1):
            row = []
            for ag in range(self.max_goals + 1):
                p = self._poisson_pmf(hg, lambda_home) * self._poisson_pmf(
                    ag, lambda_away
                )
                row.append(p)
            matrix.append(row)

        return matrix

    def probabilities_1x2(self, home_team, away_team):
        matrix = self.score_matrix(home_team, away_team)

        home_win = 0.0
        draw = 0.0
        away_win = 0.0

        for hg, row in enumerate(matrix):
            for ag, prob in enumerate(row):
                if hg > ag:
                    home_win += prob
                elif hg == ag:
                    draw += prob
                else:
                    away_win += prob

        total = home_win + draw + away_win
        if total <= 0:
            return {
                "home_win": 1 / 3,
                "draw": 1 / 3,
                "away_win": 1 / 3,
            }

        return {
            "home_win": home_win / total,
            "draw": draw / total,
            "away_win": away_win / total,
        }

    def probabilities_ou_btts(self, home_team, away_team):
        matrix = self.score_matrix(home_team, away_team)

        over_25 = 0.0
        under_25 = 0.0
        btts_yes = 0.0
        btts_no = 0.0

        for hg, row in enumerate(matrix):
            for ag, prob in enumerate(row):
                if hg + ag > 2:
                    over_25 += prob
                else:
                    under_25 += prob

                if hg > 0 and ag > 0:
                    btts_yes += prob
                else:
                    btts_no += prob

        total_ou = over_25 + under_25
        total_btts = btts_yes + btts_no

        return {
            "over_25": over_25 / total_ou if total_ou else 0.5,
            "under_25": under_25 / total_ou if total_ou else 0.5,
            "btts_yes": btts_yes / total_btts if total_btts else 0.5,
            "btts_no": btts_no / total_btts if total_btts else 0.5,
        }

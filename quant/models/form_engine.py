from __future__ import annotations


class FormEngine:

    def __init__(self, lookback=5):
        self.lookback = lookback
        self.team_form = {}

    def fit(self, completed_matches):
        history = {}

        for match in completed_matches:
            home = match["home_team"]
            away = match["away_team"]
            hg = int(match["home_goals"])
            ag = int(match["away_goals"])

            if home not in history:
                history[home] = []

            if away not in history:
                history[away] = []

            if hg > ag:
                history[home].append(1.0)
                history[away].append(0.0)
            elif hg < ag:
                history[home].append(0.0)
                history[away].append(1.0)
            else:
                history[home].append(0.5)
                history[away].append(0.5)

        team_form = {}
        for team, values in history.items():
            recent = values[-self.lookback :]
            if not recent:
                team_form[team] = 0.5
            else:
                team_form[team] = sum(recent) / len(recent)

        self.team_form = team_form

    def get_form(self, team_name):
        return self.team_form.get(team_name, 0.5)

    def get_form_diff(self, home_team, away_team):
        return self.get_form(home_team) - self.get_form(away_team)

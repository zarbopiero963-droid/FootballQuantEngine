class EloEngine:

    def __init__(self, base_rating=1500.0, k_factor=20.0, home_advantage=55.0):
        self.base_rating = base_rating
        self.k_factor = k_factor
        self.home_advantage = home_advantage
        self.ratings = {}

    def get_rating(self, team_name):
        return self.ratings.get(team_name, self.base_rating)

    def expected_score(self, rating_a, rating_b):
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))

    def update_match(self, home_team, away_team, home_goals, away_goals):
        home_rating = self.get_rating(home_team)
        away_rating = self.get_rating(away_team)

        adjusted_home = home_rating + self.home_advantage
        expected_home = self.expected_score(adjusted_home, away_rating)
        expected_away = 1.0 - expected_home

        if home_goals > away_goals:
            actual_home = 1.0
            actual_away = 0.0
        elif home_goals < away_goals:
            actual_home = 0.0
            actual_away = 1.0
        else:
            actual_home = 0.5
            actual_away = 0.5

        new_home = home_rating + self.k_factor * (actual_home - expected_home)
        new_away = away_rating + self.k_factor * (actual_away - expected_away)

        self.ratings[home_team] = new_home
        self.ratings[away_team] = new_away

    def fit(self, completed_matches):
        for match in completed_matches:
            home_team = match["home_team"]
            away_team = match["away_team"]
            home_goals = int(match["home_goals"])
            away_goals = int(match["away_goals"])

            self.update_match(
                home_team=home_team,
                away_team=away_team,
                home_goals=home_goals,
                away_goals=away_goals,
            )

    def get_elo_diff(self, home_team, away_team):
        home_rating = self.get_rating(home_team)
        away_rating = self.get_rating(away_team)
        return (home_rating + self.home_advantage) - away_rating

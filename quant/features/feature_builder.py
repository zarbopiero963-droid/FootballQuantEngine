from quant.features.advanced_stats_features import AdvancedStatsFeatureBuilder


class FeatureBuilder:

    def __init__(self, elo_engine, poisson_engine, form_engine, advanced_stats=None):
        self.elo_engine = elo_engine
        self.poisson_engine = poisson_engine
        self.form_engine = form_engine
        self.advanced_builder = AdvancedStatsFeatureBuilder(advanced_stats or {})

    def build_fixture_features(self, fixture):
        home_team = fixture["home_team"]
        away_team = fixture["away_team"]

        elo_diff = self.elo_engine.get_elo_diff(home_team, away_team)
        form_diff = self.form_engine.get_form_diff(home_team, away_team)
        lambda_home, lambda_away = self.poisson_engine.expected_goals(
            home_team, away_team
        )

        advanced = self.advanced_builder.build(home_team, away_team)

        features = {
            "fixture_id": fixture["fixture_id"],
            "home_team": home_team,
            "away_team": away_team,
            "elo_diff": elo_diff,
            "form_diff": form_diff,
            "expected_goals_home": lambda_home,
            "expected_goals_away": lambda_away,
            "goal_expectancy_diff": lambda_home - lambda_away,
        }

        features.update(advanced)
        return features

    def build_many(self, fixtures):
        return [self.build_fixture_features(fixture) for fixture in fixtures]

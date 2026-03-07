class QuantDatasetBuilder:

    def __init__(self, fixtures_provider, odds_provider, advanced_provider):
        self.fixtures_provider = fixtures_provider
        self.odds_provider = odds_provider
        self.advanced_provider = advanced_provider

    def build_training_data(self, league=None, season=None):
        completed_matches = self.fixtures_provider.get_completed_matches(
            league=league,
            season=season,
        )

        advanced = self.advanced_provider.get_team_advanced_stats(
            league=league,
            season=season,
        )

        return {
            "completed_matches": completed_matches,
            "advanced_stats": advanced,
        }

    def build_prediction_data(self, league=None, season=None):
        upcoming_matches = self.fixtures_provider.get_upcoming_matches(
            league=league,
            season=season,
        )

        fixture_ids = [match["fixture_id"] for match in upcoming_matches]
        odds = self.odds_provider.get_prematch_odds(fixture_ids)

        return {
            "upcoming_matches": upcoming_matches,
            "odds": odds,
        }

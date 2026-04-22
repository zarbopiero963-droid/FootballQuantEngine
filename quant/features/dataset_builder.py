class QuantDatasetBuilder:

    def __init__(self, fixtures_provider, odds_provider):
        self.fixtures_provider = fixtures_provider
        self.odds_provider = odds_provider

    def build_training_data(self, league=None, season=None):
        completed_matches = self.fixtures_provider.get_completed_matches(
            league=league,
            season=season,
        )

        # xG averages per team (from fixture statistics when available)
        xg_averages = {}
        if hasattr(self.fixtures_provider, "get_team_xg_averages"):
            try:
                xg_averages = (
                    self.fixtures_provider.get_team_xg_averages(
                        league=league, season=season
                    )
                    or {}
                )
            except Exception:
                pass

        standings = []
        if hasattr(self.fixtures_provider, "get_standings"):
            try:
                standings = self.fixtures_provider.get_standings(
                    league=league, season=season
                ) or []
            except Exception:
                pass

        injuries = {}
        if hasattr(self.fixtures_provider, "get_injuries"):
            try:
                injuries = (
                    self.fixtures_provider.get_injuries(
                        league=league, season=season
                    )
                    or {}
                )
            except Exception:
                pass

        referee_stats = {}
        if hasattr(self.fixtures_provider, "get_referee_stats"):
            try:
                referee_stats = (
                    self.fixtures_provider.get_referee_stats(
                        league=league, season=season
                    )
                    or {}
                )
            except Exception:
                pass

        return {
            "completed_matches": completed_matches,
            "advanced_stats": xg_averages,
            "standings": standings,
            "injuries": injuries,
            "referee_stats": referee_stats,
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

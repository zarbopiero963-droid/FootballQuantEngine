class APIFootballProvider:

    def __init__(self, client=None):
        self.client = client

    def get_completed_matches(self, league=None, season=None):
        if self.client is None:
            return []

        return self.client.get_completed_matches(
            league=league,
            season=season,
        )

    def get_upcoming_matches(self, league=None, season=None):
        if self.client is None:
            return []

        return self.client.get_upcoming_matches(
            league=league,
            season=season,
        )

    def get_prematch_odds(self, fixture_ids):
        if self.client is None:
            return {}

        return self.client.get_prematch_odds(fixture_ids=fixture_ids)

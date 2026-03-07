class UnderstatProvider:

    def __init__(self, client=None):
        self.client = client

    def get_team_advanced_stats(self, league=None, season=None):
        if self.client is None:
            return {}

        return self.client.get_team_advanced_stats(
            league=league,
            season=season,
        )

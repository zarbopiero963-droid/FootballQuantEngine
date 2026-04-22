class APIFootballProvider:

    def __init__(self, client=None):
        self.client = client

    def _call(self, method, *args, **kwargs):
        if self.client is None:
            return None
        fn = getattr(self.client, method, None)
        if fn is None:
            return None
        return fn(*args, **kwargs)

    def get_completed_matches(self, league=None, season=None):
        return self._call("get_completed_matches", league=league, season=season) or []

    def get_upcoming_matches(self, league=None, season=None):
        return self._call("get_upcoming_matches", league=league, season=season) or []

    def get_prematch_odds(self, fixture_ids):
        return self._call("get_prematch_odds", fixture_ids) or {}

    def get_standings(self, league=None, season=None):
        return self._call("get_standings", league=league, season=season) or []

    def get_injuries(self, league=None, season=None):
        return self._call("get_injuries", league=league, season=season) or {}

    def get_referee_stats(self, league=None, season=None):
        return self._call("get_referee_stats", league=league, season=season) or {}

    def get_team_xg_averages(self, league=None, season=None):
        return self._call("get_team_xg_averages", league=league, season=season) or {}

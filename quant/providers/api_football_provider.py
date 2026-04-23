from __future__ import annotations

from cache.cache_manager import TTL, load_cache, save_cache


class APIFootballProvider:
    """
    Wraps an API client with transparent JSON file caching.
    Cache keys are namespaced by league and season to avoid cross-league pollution.
    TTLs are defined in cache.cache_manager.TTL.
    """

    def __init__(self, client=None):
        self.client = client

    def _call(self, method: str, *args, **kwargs):
        if self.client is None:
            return None
        fn = getattr(self.client, method, None)
        if fn is None:
            return None
        return fn(*args, **kwargs)

    def _cached(self, cache_key: str, method: str, ttl: int | None = None, **kwargs):
        cached = load_cache(cache_key, ttl)
        if cached is not None:
            return cached
        result = self._call(method, **kwargs)
        if result is not None:
            save_cache(cache_key, result)
        return result

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_completed_matches(self, league=None, season=None) -> list:
        key = f"completed_matches_{league}_{season}"
        return (
            self._cached(key, "get_completed_matches", league=league, season=season)
            or []
        )

    def get_upcoming_matches(self, league=None, season=None) -> list:
        key = f"upcoming_matches_{league}_{season}"
        return (
            self._cached(
                key,
                "get_upcoming_matches",
                TTL["upcoming_matches"],
                league=league,
                season=season,
            )
            or []
        )

    def get_prematch_odds(self, fixture_ids) -> dict:
        key = f"odds_{'_'.join(str(i) for i in sorted(fixture_ids))}"
        cached = load_cache(key, TTL["odds"])
        if cached is not None:
            return cached
        result = self._call("get_prematch_odds", fixture_ids) or {}
        save_cache(key, result)
        return result

    def get_standings(self, league=None, season=None) -> list:
        key = f"standings_{league}_{season}"
        return self._cached(key, "get_standings", league=league, season=season) or []

    def get_injuries(self, league=None, season=None) -> dict:
        key = f"injuries_{league}_{season}"
        return self._cached(key, "get_injuries", league=league, season=season) or {}

    def get_referee_stats(self, league=None, season=None) -> dict:
        key = f"referee_stats_{league}_{season}"
        return (
            self._cached(key, "get_referee_stats", league=league, season=season) or {}
        )

    def get_team_xg_averages(self, league=None, season=None) -> dict:
        key = f"xg_averages_{league}_{season}"
        return (
            self._cached(key, "get_team_xg_averages", league=league, season=season)
            or {}
        )

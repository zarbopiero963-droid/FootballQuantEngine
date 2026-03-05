from config.constants import BASE_URL_API_FOOTBALL
from config.settings_manager import load_settings
from data.api_client import ApiClient


class ApiFootballCollector:

    def __init__(self):

        settings = load_settings()

        self.client = ApiClient(BASE_URL_API_FOOTBALL, settings.api_football_key)

    def get_fixtures(self, league_id, season):

        headers = {"x-apisports-key": load_settings().api_football_key}

        return self.client.get(
            "fixtures", params={"league": league_id, "season": season}, headers=headers
        )

    def get_next_matches(self):

        headers = {"x-apisports-key": load_settings().api_football_key}

        return self.client.get("fixtures", params={"next": 50}, headers=headers)

from config.constants import ODDS_API_BASE_URL
from config.settings_manager import load_settings
from data.api_client import ApiClient


class OddsCollector:

    def __init__(self):

        settings = load_settings()

        self.api_key = settings.odds_api_key

        self.client = ApiClient(ODDS_API_BASE_URL)

    def get_odds(self):

        return self.client.get(
            "sports/soccer/odds",
            params={"apiKey": self.api_key, "regions": "eu", "markets": "h2h"},
        )

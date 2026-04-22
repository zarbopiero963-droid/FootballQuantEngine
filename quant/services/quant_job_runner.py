from config.settings_manager import load_settings
from quant.providers.api_football_client import APIFootballClient
from quant.providers.sample_clients import SampleAPIFootballClient
from quant.services.quant_controller import QuantController


class QuantJobRunner:

    def __init__(self):
        settings = load_settings()

        if settings.api_football_key:
            api_client = APIFootballClient(api_key=settings.api_football_key)
        else:
            api_client = SampleAPIFootballClient()

        self.controller = QuantController(api_client=api_client)
        self._settings = settings

    def run_cycle(self, league=None, season=None):
        league = league if league is not None else self._settings.league_id
        season = season if season is not None else self._settings.season
        return self.controller.run(league=league, season=season)

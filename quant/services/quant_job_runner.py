from __future__ import annotations

from config.settings_manager import load_settings
from quant.providers.api_football_client import APIFootballClient
from quant.providers.sample_clients import SampleAPIFootballClient
from quant.providers.weather_client import WeatherClient, WeatherEngine
from quant.services.quant_controller import QuantController


class QuantJobRunner:

    def __init__(self):
        settings = load_settings()

        if settings.api_football_key:
            api_client = APIFootballClient(api_key=settings.api_football_key)
        else:
            api_client = SampleAPIFootballClient()

        weather_engine = None
        if settings.openweather_key:
            weather_client = WeatherClient(api_key=settings.openweather_key)
            weather_engine = WeatherEngine()
            self._weather_client = weather_client
        else:
            self._weather_client = None

        self.controller = QuantController(
            api_client=api_client,
            weather_engine=weather_engine,
        )
        self._settings = settings

    def run_cycle(self, league=None, season=None):
        league = league if league is not None else self._settings.league_id
        season = season if season is not None else self._settings.season
        return self.controller.run(league=league, season=season)

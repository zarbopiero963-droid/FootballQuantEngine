from quant.providers.api_football_provider import APIFootballProvider
from quant.providers.understat_provider import UnderstatProvider
from quant.services.quant_engine import QuantEngine
from quant.services.result_serializer import to_records


class QuantController:

    def __init__(self, api_client=None, understat_client=None):
        self.fixtures_provider = APIFootballProvider(api_client)
        self.odds_provider = APIFootballProvider(api_client)
        self.advanced_provider = UnderstatProvider(understat_client)

        self.engine = QuantEngine(
            fixtures_provider=self.fixtures_provider,
            odds_provider=self.odds_provider,
            advanced_provider=self.advanced_provider,
        )

    def run(self, league=None, season=None):
        self.engine.fit(league=league, season=season)
        results = self.engine.predict(league=league, season=season)
        return to_records(results)

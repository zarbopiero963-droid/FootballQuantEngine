from quant.providers.sample_clients import (
    SampleAPIFootballClient,
    SampleUnderstatClient,
)
from quant.services.quant_controller import QuantController


class QuantJobRunner:

    def __init__(self):
        self.controller = QuantController(
            api_client=SampleAPIFootballClient(),
            understat_client=SampleUnderstatClient(),
        )

    def run_cycle(self, league="Serie A", season=2024):
        return self.controller.run(league=league, season=season)

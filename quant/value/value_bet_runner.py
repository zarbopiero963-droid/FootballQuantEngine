from __future__ import annotations

from quant.services.quant_job_runner import QuantJobRunner
from quant.value.value_bet_engine import ValueBetEngine


class ValueBetRunner:

    def __init__(self):
        self.quant_runner = QuantJobRunner()
        self.engine = ValueBetEngine()

    def run(self, bankroll: float = 1000.0) -> list[dict]:
        records = self.quant_runner.run_cycle()
        return self.engine.run(records, bankroll=bankroll)

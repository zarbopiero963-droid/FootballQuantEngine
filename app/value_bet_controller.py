from __future__ import annotations

from quant.value.value_bet_runner import ValueBetRunner


class AppValueBetController:

    def __init__(self):
        self.runner = ValueBetRunner()

    def run(self, bankroll: float = 1000.0):
        return self.runner.run(bankroll=bankroll)

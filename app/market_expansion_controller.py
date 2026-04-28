from __future__ import annotations

from quant.markets.market_expansion_runner import MarketExpansionRunner


class AppMarketExpansionController:

    def __init__(self):
        self.runner = MarketExpansionRunner()

    def run(self, bankroll: float = 1000.0):
        return self.runner.run(bankroll=bankroll)

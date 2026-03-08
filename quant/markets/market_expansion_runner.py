from __future__ import annotations

from quant.markets.market_expansion_engine import MarketExpansionEngine


class MarketExpansionRunner:

    def __init__(self):
        self.engine = MarketExpansionEngine()

    def run(self, bankroll: float = 1000.0) -> list[dict]:
        return self.engine.run(bankroll=bankroll)

from __future__ import annotations

from quant.value.kelly_engine import KellyEngine


class StakePolicy:

    def __init__(self):
        self.kelly = KellyEngine()

    def suggest(
        self,
        bankroll: float,
        probability: float,
        odds: float,
        confidence: float,
        fraction: float = 0.25,
    ) -> dict:
        confidence = float(confidence)

        if confidence >= 0.80:
            max_cap = 0.05
        elif confidence >= 0.65:
            max_cap = 0.035
        else:
            max_cap = 0.02

        stake = self.kelly.suggested_stake(
            bankroll=bankroll,
            probability=probability,
            odds=odds,
            fraction=fraction,
            max_bankroll_pct=max_cap,
        )

        return {
            "stake": round(stake, 2),
            "max_cap_pct": max_cap,
        }

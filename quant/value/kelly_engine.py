from __future__ import annotations


class KellyEngine:

    def kelly_fraction(self, probability: float, odds: float) -> float:
        probability = float(probability)
        odds = float(odds or 0.0)

        if odds <= 1.0:
            return 0.0

        b = odds - 1.0
        q = 1.0 - probability

        value = ((b * probability) - q) / b

        return max(0.0, value)

    def fractional_kelly(
        self,
        probability: float,
        odds: float,
        fraction: float = 0.25,
    ) -> float:
        base = self.kelly_fraction(probability, odds)
        return max(0.0, base * float(fraction))

    def suggested_stake(
        self,
        bankroll: float,
        probability: float,
        odds: float,
        fraction: float = 0.25,
        max_bankroll_pct: float = 0.05,
    ) -> float:
        bankroll = float(bankroll or 0.0)
        max_bankroll_pct = float(max_bankroll_pct)

        if bankroll <= 0.0:
            return 0.0

        frac = self.fractional_kelly(
            probability=probability,
            odds=odds,
            fraction=fraction,
        )

        frac = min(frac, max_bankroll_pct)

        return round(bankroll * frac, 2)

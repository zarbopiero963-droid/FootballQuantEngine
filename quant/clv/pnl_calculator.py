from __future__ import annotations


class PnLCalculator:

    def settle(self, stake: float, odds: float, result: str) -> float:
        stake = float(stake or 0.0)
        odds = float(odds or 0.0)
        result = str(result or "").upper()

        if stake <= 0.0:
            return 0.0

        if result == "WIN":
            return round((stake * odds) - stake, 2)

        if result == "LOSE":
            return round(-stake, 2)

        if result == "VOID":
            return 0.0

        return 0.0

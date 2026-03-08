from __future__ import annotations


class CLVTracker:

    def implied_probability(self, odds: float) -> float:
        odds = float(odds or 0.0)
        if odds <= 0.0:
            return 0.0
        return 1.0 / odds

    def clv_absolute(self, placed_odds: float, closing_odds: float) -> float:
        placed_odds = float(placed_odds or 0.0)
        closing_odds = float(closing_odds or 0.0)

        if placed_odds <= 0.0 or closing_odds <= 0.0:
            return 0.0

        placed_imp = self.implied_probability(placed_odds)
        closing_imp = self.implied_probability(closing_odds)

        return placed_imp - closing_imp

    def clv_percent(self, placed_odds: float, closing_odds: float) -> float:
        placed_odds = float(placed_odds or 0.0)
        closing_odds = float(closing_odds or 0.0)

        if placed_odds <= 0.0 or closing_odds <= 0.0:
            return 0.0

        return (placed_odds - closing_odds) / placed_odds

    def evaluate(self, placed_odds: float, closing_odds: float) -> dict:
        return {
            "clv_abs": round(self.clv_absolute(placed_odds, closing_odds), 6),
            "clv_pct": round(self.clv_percent(placed_odds, closing_odds), 6),
        }

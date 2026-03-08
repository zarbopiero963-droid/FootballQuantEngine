from __future__ import annotations


class EVCalculator:

    def expected_value(self, probability: float, odds: float) -> float:
        probability = float(probability)
        odds = float(odds or 0.0)

        if odds <= 0.0:
            return 0.0

        return (probability * odds) - 1.0

    def fair_odds(self, probability: float) -> float:
        probability = float(probability)

        if probability <= 0.0:
            return 999.0

        return 1.0 / probability

    def implied_probability(self, odds: float) -> float:
        odds = float(odds or 0.0)

        if odds <= 0.0:
            return 0.0

        return 1.0 / odds

    def probability_edge(self, model_probability: float, odds: float) -> float:
        return float(model_probability) - self.implied_probability(odds)

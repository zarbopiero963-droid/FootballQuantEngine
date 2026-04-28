from __future__ import annotations

from config.constants import EDGE_THRESHOLD


class EdgeDetector:
    def calculate_edge(self, probability: float, odds: float) -> float:
        if odds <= 0:
            return 0.0
        implied_probability = 1.0 / odds
        return probability - implied_probability

    def is_value_bet(self, probability: float | None, odds: float | None) -> bool:
        if probability is None or odds is None:
            return False
        edge = self.calculate_edge(probability, odds)
        return edge > EDGE_THRESHOLD

from __future__ import annotations


class CLVEstimator:

    def estimate_clv_edge(
        self,
        model_probability: float,
        opening_odds: float,
        current_odds: float,
    ) -> float:
        model_probability = float(model_probability)
        opening_odds = float(opening_odds or 0.0)
        current_odds = float(current_odds or 0.0)

        if opening_odds <= 0.0 or current_odds <= 0.0:
            return 0.0

        opening_ev = (model_probability * opening_odds) - 1.0
        current_ev = (model_probability * current_odds) - 1.0

        return current_ev - opening_ev

    def odds_movement_score(self, opening_odds: float, current_odds: float) -> float:
        opening_odds = float(opening_odds or 0.0)
        current_odds = float(current_odds or 0.0)

        if opening_odds <= 0.0 or current_odds <= 0.0:
            return 0.0

        return (opening_odds - current_odds) / opening_odds

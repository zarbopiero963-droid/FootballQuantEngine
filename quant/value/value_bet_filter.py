from __future__ import annotations


class ValueBetFilter:

    def __init__(
        self,
        min_ev: float = 0.02,
        min_probability: float = 0.50,
        min_confidence: float = 0.60,
        min_agreement: float = 0.55,
        min_odds: float = 1.40,
        max_odds: float = 8.00,
    ):
        self.min_ev = float(min_ev)
        self.min_probability = float(min_probability)
        self.min_confidence = float(min_confidence)
        self.min_agreement = float(min_agreement)
        self.min_odds = float(min_odds)
        self.max_odds = float(max_odds)

    def decide(
        self,
        probability: float,
        odds: float,
        ev: float,
        confidence: float,
        agreement: float,
    ) -> str:
        probability = float(probability)
        odds = float(odds)
        ev = float(ev)
        confidence = float(confidence)
        agreement = float(agreement)

        if (
            probability >= self.min_probability
            and self.min_odds <= odds <= self.max_odds
            and ev >= self.min_ev
            and confidence >= self.min_confidence
            and agreement >= self.min_agreement
        ):
            return "BET"

        if (
            probability >= self.min_probability * 0.95
            and self.min_odds <= odds <= (self.max_odds * 1.25)
            and ev >= self.min_ev * 0.50
            and confidence >= self.min_confidence * 0.85
            and agreement >= self.min_agreement * 0.85
        ):
            return "WATCHLIST"

        return "NO_BET"

from __future__ import annotations


class NoBetFilter:
    def __init__(
        self,
        min_probability: float = 0.54,
        min_edge: float = 0.05,
        min_confidence: float = 0.65,
        min_agreement: float = 0.60,
    ) -> None:
        self.min_probability = min_probability
        self.min_edge = min_edge
        self.min_confidence = min_confidence
        self.min_agreement = min_agreement

    def decide(
        self,
        probability: float,
        edge: float,
        confidence: float,
        agreement: float,
    ) -> str:
        probability = float(probability)
        edge = float(edge)
        confidence = float(confidence)
        agreement = float(agreement)

        if (
            probability >= self.min_probability
            and edge >= self.min_edge
            and confidence >= self.min_confidence
            and agreement >= self.min_agreement
        ):
            return "BET"

        if (
            probability >= self.min_probability * 0.95
            and edge >= self.min_edge * 0.6
            and confidence >= self.min_confidence * 0.8
            and agreement >= self.min_agreement * 0.8
        ):
            return "WATCHLIST"

        return "NO_BET"

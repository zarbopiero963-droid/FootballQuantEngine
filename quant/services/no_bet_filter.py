class QuantNoBetFilter:

    def __init__(
        self,
        min_probability=0.54,
        min_edge=0.04,
        min_confidence=0.64,
        min_agreement=0.58,
    ):
        self.min_probability = min_probability
        self.min_edge = min_edge
        self.min_confidence = min_confidence
        self.min_agreement = min_agreement

    def decide(self, probability, edge, confidence, agreement):
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
            and edge >= self.min_edge * 0.65
            and confidence >= self.min_confidence * 0.80
            and agreement >= self.min_agreement * 0.85
        ):
            return "WATCHLIST"

        return "NO_BET"

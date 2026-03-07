class QuantConfidenceEngine:

    def score(self, probability, edge, agreement, xg_support=0.0):
        probability = max(0.0, min(1.0, float(probability)))
        edge = max(0.0, float(edge))
        agreement = max(0.0, min(1.0, float(agreement)))
        xg_support = max(0.0, min(1.0, float(xg_support)))

        edge_score = min(edge / 0.15, 1.0)

        confidence = (
            probability * 0.40
            + edge_score * 0.28
            + agreement * 0.22
            + xg_support * 0.10
        )

        confidence = max(0.0, min(1.0, confidence))
        return confidence

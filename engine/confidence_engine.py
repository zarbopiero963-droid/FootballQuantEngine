class ConfidenceEngine:

    def score(self, probability, edge, agreement):

        probability = max(0.0, min(1.0, float(probability)))
        edge = max(0.0, float(edge))
        agreement = max(0.0, min(1.0, float(agreement)))

        edge_score = min(edge / 0.15, 1.0)

        confidence = probability * 0.45 + edge_score * 0.30 + agreement * 0.25

        confidence = max(0.0, min(1.0, confidence))

        return confidence

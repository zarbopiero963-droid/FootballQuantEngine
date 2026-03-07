class QuantRanker:

    def rank(self, prediction_results):
        return sorted(
            prediction_results,
            key=lambda x: (
                x.decision != "BET",
                -x.confidence,
                -x.market_edge,
                -x.probability,
            ),
        )

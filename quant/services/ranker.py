from __future__ import annotations

from typing import Any


class QuantRanker:

    def rank(self, prediction_results: list[Any]) -> list[Any]:
        return sorted(
            prediction_results,
            key=lambda x: (
                x.decision != "BET",
                -x.confidence,
                -x.market_edge,
                -x.probability,
            ),
        )

from __future__ import annotations


class ValueBetRanker:

    def sort(self, rows: list[dict]) -> list[dict]:
        return sorted(
            rows,
            key=lambda row: (
                row.get("decision") != "BET",
                -float(row.get("ev", 0.0)),
                -float(row.get("confidence", 0.0)),
                -float(row.get("agreement", 0.0)),
                -float(row.get("clv_potential", 0.0)),
            ),
        )

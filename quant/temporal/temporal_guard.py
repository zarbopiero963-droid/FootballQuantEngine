from __future__ import annotations


class TemporalGuard:

    def __init__(
        self,
        min_rows: int = 1,
        min_bet_count: int = 0,
        min_roi: float = -1.0,
        min_yield: float = -1.0,
        min_hit_rate: float = 0.0,
        min_avg_ev: float = -1.0,
    ):
        self.min_rows = int(min_rows)
        self.min_bet_count = int(min_bet_count)
        self.min_roi = float(min_roi)
        self.min_yield = float(min_yield)
        self.min_hit_rate = float(min_hit_rate)
        self.min_avg_ev = float(min_avg_ev)

    def evaluate(self, metrics: dict) -> dict:
        checks = [
            {
                "name": "rows_minimum",
                "passed": int(metrics.get("rows", 0)) >= self.min_rows,
                "value": metrics.get("rows", 0),
            },
            {
                "name": "bet_count_minimum",
                "passed": int(metrics.get("bet_count", 0)) >= self.min_bet_count,
                "value": metrics.get("bet_count", 0),
            },
            {
                "name": "roi_threshold",
                "passed": float(metrics.get("roi", 0.0)) >= self.min_roi,
                "value": metrics.get("roi", 0.0),
            },
            {
                "name": "yield_threshold",
                "passed": float(metrics.get("yield", 0.0)) >= self.min_yield,
                "value": metrics.get("yield", 0.0),
            },
            {
                "name": "hit_rate_threshold",
                "passed": float(metrics.get("hit_rate", 0.0)) >= self.min_hit_rate,
                "value": metrics.get("hit_rate", 0.0),
            },
            {
                "name": "avg_ev_threshold",
                "passed": float(metrics.get("avg_ev", 0.0)) >= self.min_avg_ev,
                "value": metrics.get("avg_ev", 0.0),
            },
        ]

        return {
            "passed": all(item["passed"] for item in checks),
            "checks": checks,
            "metrics": metrics,
        }

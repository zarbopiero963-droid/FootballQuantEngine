from __future__ import annotations

import json
import os


class QuantGuardBaseline:

    def __init__(self, path: str = "outputs/ci/quant_guard_baseline.json"):
        self.path = path

    def load(self) -> dict | None:
        if not os.path.exists(self.path):
            return None

        with open(self.path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def save(self, summary: dict) -> str:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)

        return self.path

    def compare(self, current: dict, baseline: dict | None) -> dict:
        if not baseline:
            return {
                "has_baseline": False,
                "degraded": False,
                "details": [],
            }

        details = []

        for key in (
            "record_count",
            "bet_count",
            "avg_probability",
            "avg_confidence",
            "avg_agreement",
        ):
            current_value = current.get(key, 0)
            baseline_value = baseline.get(key, 0)

            details.append(
                {
                    "metric": key,
                    "current": current_value,
                    "baseline": baseline_value,
                    "delta": round(float(current_value) - float(baseline_value), 6),
                }
            )

        degraded = bool(current.get("record_count", 0) == 0)

        return {
            "has_baseline": True,
            "degraded": degraded,
            "details": details,
        }

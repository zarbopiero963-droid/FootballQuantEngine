from __future__ import annotations

import pandas as pd


class EdgeStabilityDetector:

    def analyze(self, rows: list[dict]) -> dict:
        if not rows:
            return {
                "count": 0,
                "avg_ev": 0.0,
                "std_ev": 0.0,
                "stable": False,
            }

        df = pd.DataFrame(rows).copy()

        if "ev" not in df.columns:
            df["ev"] = 0.0

        avg_ev = float(df["ev"].astype(float).mean())
        std_ev = float(df["ev"].astype(float).std(ddof=0))

        return {
            "count": int(len(df.index)),
            "avg_ev": round(avg_ev, 6),
            "std_ev": round(std_ev, 6),
            "stable": bool(std_ev <= max(0.05, abs(avg_ev))),
        }

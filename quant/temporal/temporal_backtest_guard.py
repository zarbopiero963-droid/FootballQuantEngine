from __future__ import annotations

import pandas as pd

from quant.temporal.temporal_guard import TemporalGuard
from quant.temporal.temporal_metrics import TemporalMetrics
from quant.temporal.temporal_reporter import TemporalReporter
from quant.temporal.time_splitter import TimeSplitter


class TemporalBacktestGuard:

    def __init__(self):
        self.splitter = TimeSplitter()
        self.metrics = TemporalMetrics()
        self.guard = TemporalGuard()
        self.reporter = TemporalReporter()

    def _ensure_date_column(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        if "match_date" not in out.columns:
            if "created_at" in out.columns:
                out["match_date"] = out["created_at"]
            else:
                out["match_date"] = pd.date_range(
                    start="2024-01-01",
                    periods=len(out.index),
                    freq="D",
                )

        return out

    def run(
        self,
        df: pd.DataFrame,
        date_col: str = "match_date",
        train_size: int = 50,
        test_size: int = 20,
        step: int = 20,
    ) -> dict:
        if df is None or df.empty:
            payload = {
                "passed": False,
                "guard": {
                    "passed": False,
                    "checks": [],
                    "metrics": self.metrics.build(pd.DataFrame()),
                },
                "windows": [],
            }
            json_path = self.reporter.save_json(payload)
            csv_path = self.reporter.save_csv([])
            html_path = self.reporter.save_html(payload)
            return {
                "passed": False,
                "payload": payload,
                "json_path": json_path,
                "csv_path": csv_path,
                "html_path": html_path,
            }

        work = self._ensure_date_column(df)
        windows = self.splitter.rolling_windows(
            work,
            date_col=date_col,
            train_size=train_size,
            test_size=test_size,
            step=step,
        )

        summary_rows = []

        for idx, window in enumerate(windows, start=1):
            test_df = window["test"]
            metrics = self.metrics.build(test_df)

            summary_rows.append(
                {
                    "window_id": idx,
                    "train_start": window["train_start"],
                    "train_end": window["train_end"],
                    "test_start": window["test_start"],
                    "test_end": window["test_end"],
                    "rows": metrics["rows"],
                    "bet_count": metrics["bet_count"],
                    "roi": metrics["roi"],
                    "yield": metrics["yield"],
                    "hit_rate": metrics["hit_rate"],
                    "avg_ev": metrics["avg_ev"],
                    "avg_clv_abs": metrics["avg_clv_abs"],
                    "avg_clv_pct": metrics["avg_clv_pct"],
                }
            )

        aggregate_df = pd.DataFrame(summary_rows)

        if aggregate_df.empty:
            aggregate_metrics = self.metrics.build(pd.DataFrame())
        else:
            aggregate_metrics = {
                "rows": int(aggregate_df["rows"].sum()),
                "bet_count": int(aggregate_df["bet_count"].sum()),
                "total_stake": 0.0,
                "total_pnl": 0.0,
                "roi": round(float(aggregate_df["roi"].mean()), 6),
                "yield": round(float(aggregate_df["yield"].mean()), 6),
                "hit_rate": round(float(aggregate_df["hit_rate"].mean()), 6),
                "avg_ev": round(float(aggregate_df["avg_ev"].mean()), 6),
                "avg_clv_abs": round(float(aggregate_df["avg_clv_abs"].mean()), 6),
                "avg_clv_pct": round(float(aggregate_df["avg_clv_pct"].mean()), 6),
            }

        guard_result = self.guard.evaluate(aggregate_metrics)

        payload = {
            "passed": guard_result["passed"],
            "guard": guard_result,
            "windows": summary_rows,
        }

        json_path = self.reporter.save_json(payload)
        csv_path = self.reporter.save_csv(summary_rows)
        html_path = self.reporter.save_html(payload)

        return {
            "passed": guard_result["passed"],
            "payload": payload,
            "json_path": json_path,
            "csv_path": csv_path,
            "html_path": html_path,
        }

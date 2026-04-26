"""
Real training pipeline that connects:
  DatasetBuilder  →  LocalAutoML  →  BacktestEngine  →  BacktestMetrics

Responsibilities
----------------
1. build_dataset()     — build the full rolling-feature matrix from DB
2. train()             — run AutoML search + cross-validation, persist best model
3. evaluate_backtest() — run historical backtest and compute all metrics
4. run_full_pipeline() — execute all three steps and return a unified report
"""

from __future__ import annotations

import json
import os
from datetime import datetime

from training.backtest_engine import BacktestEngine
from training.backtest_metrics import BacktestMetrics
from training.dataset_builder import DatasetBuilder
from training.local_automl import LocalAutoML, load_best_model

_REPORT_DIR = "outputs/training"


class TrainingManager:
    """
    Orchestrates the full offline training pipeline.

    Usage
    -----
    manager = TrainingManager()
    report  = manager.run_full_pipeline()
    # report contains: dataset_summary, automl_result, backtest_metrics, saved_at
    """

    def __init__(
        self,
        league_filter: bool = True,
        target_col: str = "target_home_win",
        n_optuna_trials: int = 30,
    ) -> None:
        self.dataset_builder = DatasetBuilder()
        self.automl = LocalAutoML()
        self.backtest_engine = BacktestEngine()
        self.backtest_metrics = BacktestMetrics()

        self.league_filter = league_filter
        self.target_col = target_col
        self.n_optuna_trials = n_optuna_trials

    # ------------------------------------------------------------------
    # Step 1 — Dataset
    # ------------------------------------------------------------------

    def build_dataset(self):
        """
        Build the feature matrix from the fixtures database.

        Returns
        -------
        pd.DataFrame with rolling pre-match features + outcome targets.
        Empty DataFrame when no completed fixtures exist.
        """
        return self.dataset_builder.build_training_dataset(
            league_filter=self.league_filter
        )

    def dataset_summary(self, df) -> dict:
        """Return a serialisable summary of the feature DataFrame."""
        if df is None or (hasattr(df, "empty") and df.empty):
            return {"rows": 0, "features": [], "leagues": [], "date_range": None}

        feat_cols = self.dataset_builder.feature_columns()
        present = [c for c in feat_cols if c in df.columns]

        leagues = (
            df["league"].dropna().unique().tolist() if "league" in df.columns else []
        )
        dates = None
        if "match_date" in df.columns and not df["match_date"].dropna().empty:
            dates = {
                "from": str(df["match_date"].min()),
                "to": str(df["match_date"].max()),
            }

        target_dist = {}
        for col in ("target_home_win", "target_draw", "target_away_win"):
            if col in df.columns:
                target_dist[col] = int(df[col].sum())

        return {
            "rows": len(df),
            "features": present,
            "n_features": len(present),
            "leagues": leagues,
            "date_range": dates,
            "target_distribution": target_dist,
        }

    # ------------------------------------------------------------------
    # Step 2 — AutoML
    # ------------------------------------------------------------------

    def train(self, dataset_df=None) -> dict:
        """
        Run AutoML search on the provided (or freshly built) dataset.

        Parameters
        ----------
        dataset_df : pd.DataFrame | None
            If None, calls build_dataset() automatically.

        Returns
        -------
        AutoML result dict (see LocalAutoML.run docstring).
        """
        if dataset_df is None:
            dataset_df = self.build_dataset()

        return self.automl.run(
            dataset_df,
            target_col=self.target_col,
            n_optuna_trials=self.n_optuna_trials,
        )

    # ------------------------------------------------------------------
    # Step 3 — Backtest
    # ------------------------------------------------------------------

    def evaluate_backtest(
        self,
        date_from: str | None = None,
        date_to: str | None = None,
        league: str | None = None,
    ) -> dict:
        """
        Load historical predictions from the DB and compute all metrics.

        Returns
        -------
        BacktestMetrics.calculate() result dict, or {"status": "no_data"}
        when the predictions table is empty.
        """
        df = self.backtest_engine.run(
            date_from=date_from,
            date_to=date_to,
            league=league,
        )
        if df is None or (hasattr(df, "empty") and df.empty):
            return {"status": "no_data"}

        metrics = self.backtest_metrics.calculate(df)
        metrics["status"] = "ok"
        metrics["n_predictions"] = len(df)
        return metrics

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def run_full_pipeline(
        self,
        date_from: str | None = None,
        date_to: str | None = None,
        league: str | None = None,
    ) -> dict:
        """
        Execute all three steps and return a unified training report.

        The report is also saved as JSON in outputs/training/.

        Returns
        -------
        dict with keys:
          dataset_summary   — row count, features, leagues, date range
          automl_result     — best model, log-loss, all CV results, feature importances
          backtest_metrics  — ROI, hit-rate, Brier, log-loss, drawdown, bankroll history
          saved_at          — ISO timestamp of this run
          report_path       — path to the saved JSON report
        """
        saved_at = datetime.utcnow().isoformat() + "Z"

        # 1. Dataset
        dataset_df = self.build_dataset()
        ds_summary = self.dataset_summary(dataset_df)

        # 2. AutoML
        automl_result = self.train(dataset_df)

        # 3. Backtest (from DB predictions — independent of AutoML)
        bt_metrics = self.evaluate_backtest(
            date_from=date_from,
            date_to=date_to,
            league=league,
        )

        report = {
            "saved_at": saved_at,
            "dataset_summary": ds_summary,
            "automl_result": _serialisable(automl_result),
            "backtest_metrics": _serialisable(bt_metrics),
        }

        report_path = self._save_report(report, saved_at)
        report["report_path"] = report_path

        return report

    def load_best_model(self):
        """Load the persisted best AutoML model from disk."""
        return load_best_model()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _save_report(self, report: dict, saved_at: str) -> str:
        os.makedirs(_REPORT_DIR, exist_ok=True)
        ts = saved_at.replace(":", "-").replace(".", "-")
        path = os.path.join(_REPORT_DIR, f"training_report_{ts}.json")
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(report, fh, indent=2, default=str)
        except Exception:
            pass
        return path


def _serialisable(obj):
    """Recursively convert non-JSON-serialisable values to strings."""
    if isinstance(obj, dict):
        return {k: _serialisable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialisable(v) for v in obj]
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        return str(obj)

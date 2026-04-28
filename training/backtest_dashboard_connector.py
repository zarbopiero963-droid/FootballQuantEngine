from __future__ import annotations

from utils.dashboard_metrics_loader import compute_dashboard_metrics
from utils.run_history_manager import RunHistoryManager


class BacktestDashboardConnector:

    def __init__(self):
        self.history = RunHistoryManager()

    def process_backtest(self, df):

        metrics = compute_dashboard_metrics(df)

        self.history.add_run(metrics)

        return metrics

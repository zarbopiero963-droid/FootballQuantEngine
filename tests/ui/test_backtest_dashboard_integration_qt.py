import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

try:
    from PySide6.QtCore import Qt

    from ui.backtest_runner_window import BacktestRunnerWindow
    from ui.dashboard_view import DashboardView
except Exception as exc:
    pytest.skip(
        f"Qt UI not available in this environment: {exc}",
        allow_module_level=True,
    )


class MetricsReceiver:
    def __init__(self, dashboard):
        self.dashboard = dashboard
        self.logs = []

    def on_metrics(self, metrics):
        self.dashboard.set_metrics(metrics)

    def on_log(self, message):
        self.logs.append(message)
        self.dashboard.append_log(message)


class DummyBacktestRunnerWindow(BacktestRunnerWindow):
    def run_backtest(self):
        metrics = {
            "roi": 0.12,
            "yield": 0.12,
            "hit_rate": 0.58,
            "total_profit": 12.5,
            "total_staked": 100.0,
            "total_bets": 25,
            "max_drawdown": -0.08,
            "brier_score": 0.19,
            "log_loss": 0.62,
            "bankroll_history": [100, 102, 101, 105, 112.5],
            "accuracy_history": [1.0, 0.5, 0.66, 0.75, 0.8],
            "drawdown_history": [0.0, 0.0, -0.01, 0.0, 0.0],
        }

        pretty_lines = [
            f"Total Bets: {metrics.get('total_bets', 0)}",
            f"ROI: {metrics.get('roi', 0):.2%}",
            f"Yield: {metrics.get('yield', 0):.2%}",
            f"Hit Rate: {metrics.get('hit_rate', 0):.2%}",
            f"Total Profit: {metrics.get('total_profit', 0):.2f}",
            f"Max Drawdown: {metrics.get('max_drawdown', 0):.2%}",
            f"Brier Score: {metrics.get('brier_score', 0):.4f}",
            f"Log Loss: {metrics.get('log_loss', 0):.4f}",
        ]
        self.output.setPlainText("\n".join(pretty_lines))

        if callable(self.on_metrics_ready):
            self.on_metrics_ready(metrics)

        if callable(self.on_log_message):
            self.on_log_message("Backtest completed successfully.")


def test_backtest_runner_updates_dashboard(qtbot):
    dashboard = DashboardView()
    qtbot.addWidget(dashboard)
    dashboard.show()

    receiver = MetricsReceiver(dashboard)

    window = DummyBacktestRunnerWindow(
        on_metrics_ready=receiver.on_metrics,
        on_log_message=receiver.on_log,
    )
    qtbot.addWidget(window)
    window.show()

    qtbot.mouseClick(window.run_button, Qt.LeftButton)

    qtbot.waitUntil(
        lambda: "ROI:" in dashboard.metrics_output.toPlainText(),
        timeout=5000,
    )

    assert "ROI:" in dashboard.metrics_output.toPlainText()
    assert "Backtest completed successfully." in dashboard.log_output.toPlainText()
    assert dashboard.roi_card.value_label.text() != "0.00"


def test_backtest_runner_output_is_filled(qtbot):
    dashboard = DashboardView()
    qtbot.addWidget(dashboard)

    receiver = MetricsReceiver(dashboard)

    window = DummyBacktestRunnerWindow(
        on_metrics_ready=receiver.on_metrics,
        on_log_message=receiver.on_log,
    )
    qtbot.addWidget(window)
    window.show()

    qtbot.mouseClick(window.run_button, Qt.LeftButton)

    qtbot.waitUntil(
        lambda: "Total Bets:" in window.output.toPlainText(),
        timeout=5000,
    )

    assert "Total Bets:" in window.output.toPlainText()

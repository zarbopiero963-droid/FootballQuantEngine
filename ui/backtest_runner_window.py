from PySide6.QtWidgets import (
    QGridLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QWidget,
)

from training.backtest_engine import BacktestEngine
from training.backtest_metrics import BacktestMetrics


class BacktestRunnerWindow(QWidget):

    def __init__(self, on_metrics_ready=None, on_log_message=None):

        super().__init__()

        self.setWindowTitle("Backtest Runner")

        self.engine = BacktestEngine()
        self.metrics_engine = BacktestMetrics()

        self.on_metrics_ready = on_metrics_ready
        self.on_log_message = on_log_message

        self.date_from_input = QLineEdit()
        self.date_to_input = QLineEdit()
        self.league_input = QLineEdit()

        self.date_from_input.setPlaceholderText("YYYY-MM-DD")
        self.date_to_input.setPlaceholderText("YYYY-MM-DD")
        self.league_input.setPlaceholderText("Example: Serie A")

        self.run_button = QPushButton("Run Backtest")
        self.run_button.clicked.connect(self.run_backtest)

        self.output = QTextEdit()
        self.output.setReadOnly(True)

        layout = QGridLayout()
        layout.addWidget(QLabel("Date From"), 0, 0)
        layout.addWidget(self.date_from_input, 0, 1)

        layout.addWidget(QLabel("Date To"), 1, 0)
        layout.addWidget(self.date_to_input, 1, 1)

        layout.addWidget(QLabel("League"), 2, 0)
        layout.addWidget(self.league_input, 2, 1)

        layout.addWidget(self.run_button, 3, 0, 1, 2)
        layout.addWidget(QLabel("Backtest Summary"), 4, 0, 1, 2)
        layout.addWidget(self.output, 5, 0, 1, 2)

        self.setLayout(layout)

    def run_backtest(self):

        try:
            date_from = self.date_from_input.text().strip() or None
            date_to = self.date_to_input.text().strip() or None
            league = self.league_input.text().strip() or None

            df = self.engine.run(
                date_from=date_from,
                date_to=date_to,
                league=league,
            )

            metrics = self.metrics_engine.calculate(df)

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

            QMessageBox.information(
                self,
                "Backtest",
                "Backtest completed and dashboard updated.",
            )

        except Exception as exc:
            if callable(self.on_log_message):
                self.on_log_message(f"Backtest error: {exc}")

            QMessageBox.critical(self, "Backtest Error", str(exc))

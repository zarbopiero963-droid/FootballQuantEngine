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

        self.run_button = QPushButton("Run Backtest")
        self.run_button.clicked.connect(self.run_backtest)

        self.output = QTextEdit()
        self.output.setReadOnly(True)

        layout = QGridLayout()
        layout.addWidget(QLabel("Date From (YYYY-MM-DD)"), 0, 0)
        layout.addWidget(self.date_from_input, 0, 1)

        layout.addWidget(QLabel("Date To (YYYY-MM-DD)"), 1, 0)
        layout.addWidget(self.date_to_input, 1, 1)

        layout.addWidget(QLabel("League"), 2, 0)
        layout.addWidget(self.league_input, 2, 1)

        layout.addWidget(self.run_button, 3, 0, 1, 2)
        layout.addWidget(self.output, 4, 0, 1, 2)

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

            self.output.setPlainText(str(metrics))

            if callable(self.on_metrics_ready):
                self.on_metrics_ready(metrics)

            if callable(self.on_log_message):
                self.on_log_message("Backtest completed successfully.")

        except Exception as exc:
            if callable(self.on_log_message):
                self.on_log_message(f"Backtest error: {exc}")

            QMessageBox.critical(self, "Backtest Error", str(exc))

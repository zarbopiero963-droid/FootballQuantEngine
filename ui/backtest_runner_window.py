from __future__ import annotations

import glob
import os
import webbrowser

from PySide6.QtWidgets import (
    QGridLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QWidget,
)

from engine.threshold_optimizer import ThresholdOptimizer
from training.backtest_engine import BacktestEngine
from training.backtest_metrics import BacktestMetrics

_OUTPUTS_DIR = "outputs"


class BacktestRunnerWindow(QWidget):
    def __init__(self, on_metrics_ready=None, on_log_message=None):

        super().__init__()

        self.setWindowTitle("Backtest — Football Quant Engine")
        self.resize(540, 480)

        self.engine = BacktestEngine()
        self.metrics_engine = BacktestMetrics()
        self.threshold_optimizer = ThresholdOptimizer()

        self.on_metrics_ready = on_metrics_ready
        self.on_log_message = on_log_message

        self._last_backtest_df = None

        self.date_from_input = QLineEdit()
        self.date_to_input = QLineEdit()
        self.league_input = QLineEdit()

        self.date_from_input.setPlaceholderText("YYYY-MM-DD  (leave blank = all)")
        self.date_to_input.setPlaceholderText("YYYY-MM-DD  (leave blank = all)")
        self.league_input.setPlaceholderText("e.g. Serie A  (leave blank = all)")

        self.run_button = QPushButton("▶  Run Backtest")
        self.run_button.clicked.connect(self.run_backtest)

        self.optimise_button = QPushButton("🔍  Optimise Thresholds")
        self.optimise_button.setToolTip(
            "Grid-search optimal edge / confidence / kelly thresholds on the last backtest run"
        )
        self.optimise_button.setEnabled(False)
        self.optimise_button.clicked.connect(self.optimise_thresholds)

        self.open_report_button = QPushButton("📄  Open HTML Report")
        self.open_report_button.setToolTip(
            "Open the latest temporal HTML report from outputs/"
        )
        self.open_report_button.clicked.connect(self.open_html_report)

        self.output = QTextEdit()
        self.output.setReadOnly(True)
        self.output.setFontFamily("Courier New")

        layout = QGridLayout()
        layout.setSpacing(8)

        layout.addWidget(QLabel("Date From"), 0, 0)
        layout.addWidget(self.date_from_input, 0, 1, 1, 2)

        layout.addWidget(QLabel("Date To"), 1, 0)
        layout.addWidget(self.date_to_input, 1, 1, 1, 2)

        layout.addWidget(QLabel("League"), 2, 0)
        layout.addWidget(self.league_input, 2, 1, 1, 2)

        layout.addWidget(self.run_button, 3, 0)
        layout.addWidget(self.optimise_button, 3, 1)
        layout.addWidget(self.open_report_button, 3, 2)

        layout.addWidget(QLabel("Backtest Summary"), 4, 0, 1, 3)
        layout.addWidget(self.output, 5, 0, 1, 3)

        self.setLayout(layout)

    def run_backtest(self):

        self.run_button.setEnabled(False)
        self.run_button.setText("Running…")

        try:
            date_from = self.date_from_input.text().strip() or None
            date_to = self.date_to_input.text().strip() or None
            league = self.league_input.text().strip() or None

            df = self.engine.run(
                date_from=date_from,
                date_to=date_to,
                league=league,
            )

            self._last_backtest_df = df

            if df is None or df.empty:
                self.output.setPlainText(
                    "No backtest data found for the selected filters.\n"
                    "Ensure predictions exist in the database for finished matches."
                )
                if callable(self.on_log_message):
                    self.on_log_message("Backtest: no data found for selected filters.")
                return

            metrics = self.metrics_engine.calculate(df)

            self._display_metrics(metrics, len(df))

            if callable(self.on_metrics_ready):
                self.on_metrics_ready(metrics)

            if callable(self.on_log_message):
                self.on_log_message(
                    f"Backtest completed: {metrics.get('total_bets', 0)} bets, "
                    f"ROI={metrics.get('roi', 0):.2%}"
                )

            self.optimise_button.setEnabled(True)

        except Exception as exc:
            self.output.setPlainText(f"Error: {exc}")
            if callable(self.on_log_message):
                self.on_log_message(f"Backtest error: {exc}")
            QMessageBox.critical(self, "Backtest Error", str(exc))

        finally:
            self.run_button.setEnabled(True)
            self.run_button.setText("▶  Run Backtest")

    def optimise_thresholds(self):
        if self._last_backtest_df is None or self._last_backtest_df.empty:
            QMessageBox.warning(self, "Optimise", "Run a backtest first.")
            return

        try:
            self.optimise_button.setEnabled(False)
            self.optimise_button.setText("Optimising…")

            result = self.threshold_optimizer.optimise(self._last_backtest_df)

            lines = [
                "─" * 42,
                "  THRESHOLD OPTIMISER RESULTS",
                "─" * 42,
            ]
            if result.n_valid_combinations == 0:
                lines.append(
                    f"  Not enough data (need ≥{self.threshold_optimizer._min_sample} bets per combo)."
                )
                lines.append("  Keeping default thresholds.")
            else:
                t = result.best_thresholds
                lines += [
                    f"  Combinations tried : {result.n_combinations_tried}",
                    f"  Valid combinations : {result.n_valid_combinations}",
                    "",
                    "  ── Optimal thresholds ──",
                    f"  Min Edge            : {t['min_edge']:.2f}",
                    f"  Min Confidence      : {t['min_confidence']:.2f}",
                    f"  Min Kelly Fraction  : {t['min_kelly']:.2f}",
                    "",
                    "  ── Performance at optimal ──",
                    f"  Bets                : {result.best_bets}",
                    f"  ROI                 : {result.best_roi:.2%}",
                    f"  Hit Rate            : {result.best_hit_rate:.2%}",
                    f"  Max Drawdown        : {result.best_max_drawdown:.2%}",
                    "",
                    "  ── Top 3 alternatives ──",
                ]
                for rank, row in enumerate(result.all_results[1:4], start=2):
                    lines.append(
                        f"  #{rank}  edge={row['min_edge']:.2f}  conf={row['min_confidence']:.2f}"
                        f"  kelly={row['min_kelly']:.2f}  ROI={row['roi']:.2%}  bets={row['bets']}"
                    )

            lines.append("─" * 42)
            current = self.output.toPlainText()
            self.output.setPlainText(current + "\n\n" + "\n".join(lines))

            if callable(self.on_log_message):
                self.on_log_message(result.summary())

        except Exception as exc:
            QMessageBox.critical(self, "Optimise Error", str(exc))
        finally:
            self.optimise_button.setEnabled(True)
            self.optimise_button.setText("🔍  Optimise Thresholds")

    def open_html_report(self):
        """Open the most recent HTML report from the outputs/ directory."""
        pattern = os.path.join(_OUTPUTS_DIR, "*.html")
        html_files = glob.glob(pattern)

        if not html_files:
            QMessageBox.information(
                self,
                "No Report Found",
                f"No HTML reports found in '{_OUTPUTS_DIR}/'.\n"
                "Run the Analytics dashboard or temporal backtest to generate one.",
            )
            return

        # Pick the most recently modified HTML file
        latest = max(html_files, key=os.path.getmtime)
        abs_path = os.path.abspath(latest)
        url = f"file:///{abs_path.replace(os.sep, '/')}"
        webbrowser.open(url)

        if callable(self.on_log_message):
            self.on_log_message(f"Opened report: {latest}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _display_metrics(self, metrics: dict, n_rows: int) -> None:
        roi = metrics.get("roi", 0.0)
        roi_sign = "+" if roi >= 0 else ""

        lines = [
            "─" * 42,
            "  BACKTEST RESULTS",
            "─" * 42,
            f"  Total Bets          : {metrics.get('total_bets', 0)}",
            f"  Total Staked        : {metrics.get('total_staked', 0):.2f} units",
            f"  Total Profit        : {metrics.get('total_profit', 0):+.2f} units",
            f"  ROI                 : {roi_sign}{roi:.2%}",
            f"  Yield               : {roi_sign}{metrics.get('yield', 0):.2%}",
            f"  Hit Rate            : {metrics.get('hit_rate', 0):.2%}",
            f"  Max Drawdown        : {metrics.get('max_drawdown', 0):.2%}",
            "─" * 42,
            "  MODEL QUALITY",
            "─" * 42,
            f"  Brier Score         : {metrics.get('brier_score', 0):.4f}  (↓ better)",
            f"  Log Loss            : {metrics.get('log_loss', 0):.4f}  (↓ better)",
            "─" * 42,
            f"  Rows in dataset     : {n_rows}",
            "─" * 42,
        ]
        self.output.setPlainText("\n".join(lines))

import glob
import os
import webbrowser

from PySide6.QtWidgets import (
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QToolBar,
)

from ui.about_dialog import AboutDialog
from ui.backtest_runner_window import BacktestRunnerWindow
from ui.dashboard_view import DashboardView
from ui.final_check_window import FinalCheckWindow
from ui.manual_context_window import ManualContextWindow
from ui.offline_import_window import OfflineImportWindow
from ui.outputs_view import OutputsView
from ui.project_summary_window import ProjectSummaryWindow
from ui.settings_window import SettingsWindow

_OUTPUTS_DIR = "outputs"


class MainWindow(QMainWindow):
    def __init__(self, controller):

        super().__init__()

        self.controller = controller

        self.setWindowTitle("Football Quant Engine")
        self.resize(1320, 860)

        self.dashboard = DashboardView()
        self.setCentralWidget(self.dashboard)

        self.status_label = QLabel("Ready")
        self.statusBar().addPermanentWidget(self.status_label)

        toolbar = QToolBar("Main")
        self.addToolBar(toolbar)

        self.run_button = QPushButton("Run Cycle")
        self.run_button.clicked.connect(self.run_cycle)
        toolbar.addWidget(self.run_button)

        self.backtest_button = QPushButton("Backtest")
        self.backtest_button.clicked.connect(self.open_backtest)
        toolbar.addWidget(self.backtest_button)

        self.settings_button = QPushButton("Settings")
        self.settings_button.clicked.connect(self.open_settings)
        toolbar.addWidget(self.settings_button)

        self.offline_import_button = QPushButton("Import CSV")
        self.offline_import_button.clicked.connect(self.open_offline_import)
        toolbar.addWidget(self.offline_import_button)

        self.manual_context_button = QPushButton("Manual Context")
        self.manual_context_button.clicked.connect(self.open_manual_context)
        toolbar.addWidget(self.manual_context_button)

        self.outputs_button = QPushButton("Outputs")
        self.outputs_button.clicked.connect(self.open_outputs)
        toolbar.addWidget(self.outputs_button)

        self.final_check_button = QPushButton("Final Check")
        self.final_check_button.clicked.connect(self.open_final_check)
        toolbar.addWidget(self.final_check_button)

        self.project_summary_button = QPushButton("Project Summary")
        self.project_summary_button.clicked.connect(self.open_project_summary)
        toolbar.addWidget(self.project_summary_button)

        self.analytics_button = QPushButton("Analytics")
        self.analytics_button.clicked.connect(self.open_analytics)
        toolbar.addWidget(self.analytics_button)

        self.reports_button = QPushButton("HTML Report")
        self.reports_button.setToolTip("Open latest HTML report from outputs/")
        self.reports_button.clicked.connect(self.open_latest_report)
        toolbar.addWidget(self.reports_button)

        self.about_button = QPushButton("About")
        self.about_button.clicked.connect(self.open_about)
        toolbar.addWidget(self.about_button)

        self.settings_window = None
        self.about_dialog = None
        self.offline_import_window = None
        self.manual_context_window = None
        self.outputs_view = None
        self.final_check_window = None
        self.project_summary_window = None
        self.backtest_window = None
        self.analytics_window = None
        self.reports_button_ref = None

        self.dashboard.set_status("Ready")
        self.dashboard.append_log("Application started.")
        self.dashboard.add_history_message("Application started.")

    def open_settings(self):

        self.settings_window = SettingsWindow()
        self.settings_window.show()

    def open_about(self):

        self.about_dialog = AboutDialog()
        self.about_dialog.show()

    def open_offline_import(self):

        self.offline_import_window = OfflineImportWindow()
        self.offline_import_window.show()

    def open_manual_context(self):

        self.manual_context_window = ManualContextWindow()
        self.manual_context_window.show()

    def open_outputs(self):

        self.outputs_view = OutputsView()
        self.outputs_view.show()

    def open_final_check(self):

        self.final_check_window = FinalCheckWindow()
        self.final_check_window.show()

    def open_project_summary(self):

        self.project_summary_window = ProjectSummaryWindow()
        self.project_summary_window.show()

    def open_analytics(self):
        from dashboard.analytics_dashboard import AnalyticsDashboard

        self.analytics_window = AnalyticsDashboard(controller=self.controller)
        self.analytics_window.load_from_controller()
        self.analytics_window.show()

    def open_latest_report(self):
        """Open the most recently modified HTML file from outputs/."""
        html_files = glob.glob(os.path.join(_OUTPUTS_DIR, "*.html"))
        if not html_files:
            QMessageBox.information(
                self,
                "No Reports",
                f"No HTML reports found in '{_OUTPUTS_DIR}/'.\n"
                "Run Backtest or Analytics to generate a report first.",
            )
            return
        latest = max(html_files, key=os.path.getmtime)
        abs_path = os.path.abspath(latest)
        url = f"file:///{abs_path.replace(os.sep, '/')}"
        webbrowser.open(url)
        self.dashboard.append_log(f"Opened report: {latest}")

    def open_backtest(self):

        self.backtest_window = BacktestRunnerWindow(
            on_metrics_ready=self.dashboard.set_metrics,
            on_log_message=self.dashboard.append_log,
        )
        self.backtest_window.show()

    def run_cycle(self):

        self.dashboard.set_status("Running")
        self.status_label.setText("Running")
        self.dashboard.append_log("Quant prediction pipeline started.")
        self.dashboard.add_history_message("Quant prediction pipeline started.")

        try:
            ranked = self.controller.run_predictions()

            if ranked is None:
                self.dashboard.clear_table()
                self.dashboard.set_log_text("No prediction results returned.")
                self.dashboard.set_status("Completed")
                self.status_label.setText("Completed")
                self.dashboard.add_history_message(
                    "Prediction run completed: no results"
                )
                return

            if isinstance(ranked, list):
                self.dashboard.set_results_from_records(ranked)
                self.dashboard.set_log_text(str(ranked))
                self.dashboard.set_status("Completed")
                self.status_label.setText("Completed")
                self.dashboard.add_history_message(
                    f"Prediction run completed: rows={len(ranked)}"
                )
                self.dashboard.append_log(
                    f"Quant prediction run completed successfully with {len(ranked)} rows."
                )
                return

            if getattr(ranked, "empty", False):
                self.dashboard.clear_table()
                self.dashboard.set_log_text("No prediction rows found.")
                self.dashboard.set_status("Completed")
                self.status_label.setText("Completed")
                self.dashboard.add_history_message(
                    "Prediction run completed: empty dataframe"
                )
                return

            self.dashboard.set_results_dataframe(ranked)
            self.dashboard.set_log_text(ranked.to_string(index=False))
            self.dashboard.set_status("Completed")
            self.status_label.setText("Completed")
            self.dashboard.add_history_message(
                f"Prediction run completed: rows={len(ranked.index)}"
            )
            self.dashboard.append_log("Quant prediction run completed successfully.")

        except Exception as exc:
            self.dashboard.set_status("Error")
            self.dashboard.append_log(f"Error: {exc}")
            self.dashboard.add_history_message(f"Prediction run failed: {exc}")
            QMessageBox.critical(
                self,
                "Run Error",
                str(exc),
            )
            self.status_label.setText("Error")

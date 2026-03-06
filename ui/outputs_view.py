from PySide6.QtWidgets import (
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from analysis.report_viewer_helper import ReportViewerHelper
from engine.offline_controller import OfflineController


class OutputsView(QWidget):

    def __init__(self):

        super().__init__()

        self.setWindowTitle("Outputs & Reports")

        self.viewer = ReportViewerHelper()
        self.offline_controller = OfflineController()

        self.run_reports_button = QPushButton("Generate Offline Reports")
        self.open_dashboard_button = QPushButton("Open Dashboard HTML")
        self.open_report_button = QPushButton("Open Advanced Report HTML")

        self.run_reports_button.clicked.connect(self.run_reports)
        self.open_dashboard_button.clicked.connect(self.open_dashboard)
        self.open_report_button.clicked.connect(self.open_report)

        layout = QVBoxLayout()
        layout.addWidget(self.run_reports_button)
        layout.addWidget(self.open_dashboard_button)
        layout.addWidget(self.open_report_button)

        self.setLayout(layout)

    def run_reports(self):

        try:
            self.offline_controller.run_reports()

            QMessageBox.information(
                self,
                "Reports",
                "Offline reports generated successfully.",
            )
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Reports Error",
                str(exc),
            )

    def open_dashboard(self):

        ok = self.viewer.open_dashboard()

        if not ok:
            QMessageBox.warning(
                self,
                "Dashboard",
                "outputs/dashboard.html not found.",
            )

    def open_report(self):

        ok = self.viewer.open_advanced_report()

        if not ok:
            QMessageBox.warning(
                self,
                "Advanced Report",
                "outputs/advanced_report.html not found.",
            )

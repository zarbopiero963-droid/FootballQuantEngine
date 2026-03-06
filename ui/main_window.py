from PySide6.QtWidgets import (
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QToolBar,
)

from ui.about_dialog import AboutDialog
from ui.dashboard_view import DashboardView
from ui.settings_window import SettingsWindow


class MainWindow(QMainWindow):

    def __init__(self, controller):

        super().__init__()

        self.controller = controller

        self.setWindowTitle("Football Quant Engine")
        self.resize(1000, 700)

        self.dashboard = DashboardView()
        self.setCentralWidget(self.dashboard)

        self.status_label = QLabel("Ready")
        self.statusBar().addPermanentWidget(self.status_label)

        toolbar = QToolBar("Main")
        self.addToolBar(toolbar)

        self.run_button = QPushButton("Run Cycle")
        self.run_button.clicked.connect(self.run_cycle)
        toolbar.addWidget(self.run_button)

        self.settings_button = QPushButton("Settings")
        self.settings_button.clicked.connect(self.open_settings)
        toolbar.addWidget(self.settings_button)

        self.about_button = QPushButton("About")
        self.about_button.clicked.connect(self.open_about)
        toolbar.addWidget(self.about_button)

        self.settings_window = None
        self.about_dialog = None

    def open_settings(self):

        self.settings_window = SettingsWindow()
        self.settings_window.show()

    def open_about(self):

        self.about_dialog = AboutDialog()
        self.about_dialog.show()

    def run_cycle(self):

        try:
            ranked = self.controller.run_once()

            if ranked is None:
                self.dashboard.set_text("No results returned.")
                self.status_label.setText("Completed")
                return

            if getattr(ranked, "empty", False):
                self.dashboard.set_text("No value bets found.")
                self.status_label.setText("Completed")
                return

            self.dashboard.set_text(ranked.to_string(index=False))
            self.status_label.setText("Completed")

        except Exception as exc:
            QMessageBox.critical(
                self,
                "Run Error",
                str(exc),
            )
            self.status_label.setText("Error")

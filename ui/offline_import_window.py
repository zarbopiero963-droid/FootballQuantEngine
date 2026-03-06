from PySide6.QtWidgets import (
    QFileDialog,
    QGridLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QWidget,
)

from engine.offline_controller import OfflineController


class OfflineImportWindow(QWidget):

    def __init__(self):

        super().__init__()

        self.setWindowTitle("Offline CSV Import")

        self.controller = OfflineController()

        self.matches_path = QLineEdit()
        self.odds_path = QLineEdit()

        self.matches_browse = QPushButton("Browse Matches CSV")
        self.odds_browse = QPushButton("Browse Odds CSV")
        self.import_button = QPushButton("Import Data")

        self.matches_browse.clicked.connect(self.browse_matches)
        self.odds_browse.clicked.connect(self.browse_odds)
        self.import_button.clicked.connect(self.import_data)

        layout = QGridLayout()
        layout.addWidget(QLabel("Matches CSV"), 0, 0)
        layout.addWidget(self.matches_path, 0, 1)
        layout.addWidget(self.matches_browse, 0, 2)

        layout.addWidget(QLabel("Odds CSV"), 1, 0)
        layout.addWidget(self.odds_path, 1, 1)
        layout.addWidget(self.odds_browse, 1, 2)

        layout.addWidget(self.import_button, 2, 0, 1, 3)

        self.setLayout(layout)

    def browse_matches(self):

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Matches CSV",
            "",
            "CSV Files (*.csv)",
        )

        if path:
            self.matches_path.setText(path)

    def browse_odds(self):

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Odds CSV",
            "",
            "CSV Files (*.csv)",
        )

        if path:
            self.odds_path.setText(path)

    def import_data(self):

        try:
            matches_csv = self.matches_path.text().strip() or None
            odds_csv = self.odds_path.text().strip() or None

            self.controller.import_csv_data(
                matches_csv=matches_csv,
                odds_csv=odds_csv,
            )

            QMessageBox.information(
                self,
                "Import",
                "CSV import completed successfully.",
            )
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Import Error",
                str(exc),
            )

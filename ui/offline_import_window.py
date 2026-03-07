from PySide6.QtWidgets import (
    QFileDialog,
    QGridLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QWidget,
)

from engine.offline_controller import OfflineController
from training.import_validator import ImportValidator


class OfflineImportWindow(QWidget):

    def __init__(self):

        super().__init__()

        self.setWindowTitle("Offline CSV Import Wizard")

        self.controller = OfflineController()
        self.validator = ImportValidator()

        self.matches_path = QLineEdit()
        self.odds_path = QLineEdit()

        self.matches_browse = QPushButton("Browse Matches CSV")
        self.odds_browse = QPushButton("Browse Odds CSV")
        self.preview_button = QPushButton("Preview CSV")
        self.import_button = QPushButton("Validate + Import")

        self.output = QTextEdit()
        self.output.setReadOnly(True)

        self.preview_table = QTableWidget()
        self.preview_table.setRowCount(0)
        self.preview_table.setColumnCount(0)

        self.matches_browse.clicked.connect(self.browse_matches)
        self.odds_browse.clicked.connect(self.browse_odds)
        self.preview_button.clicked.connect(self.preview_data)
        self.import_button.clicked.connect(self.import_data)

        layout = QGridLayout()
        layout.addWidget(QLabel("Matches CSV"), 0, 0)
        layout.addWidget(self.matches_path, 0, 1)
        layout.addWidget(self.matches_browse, 0, 2)

        layout.addWidget(QLabel("Odds CSV"), 1, 0)
        layout.addWidget(self.odds_path, 1, 1)
        layout.addWidget(self.odds_browse, 1, 2)

        layout.addWidget(self.preview_button, 2, 0, 1, 1)
        layout.addWidget(self.import_button, 2, 1, 1, 2)

        layout.addWidget(QLabel("Preview"), 3, 0, 1, 3)
        layout.addWidget(self.preview_table, 4, 0, 1, 3)

        layout.addWidget(QLabel("Wizard Log"), 5, 0, 1, 3)
        layout.addWidget(self.output, 6, 0, 1, 3)

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

    def _render_preview(self, df):

        if df is None or df.empty:
            self.preview_table.setRowCount(0)
            self.preview_table.setColumnCount(0)
            return

        preview_df = df.head(10).copy()

        self.preview_table.setColumnCount(len(preview_df.columns))
        self.preview_table.setRowCount(len(preview_df.index))
        self.preview_table.setHorizontalHeaderLabels(list(preview_df.columns))

        for row_index in range(len(preview_df.index)):
            for col_index, col_name in enumerate(preview_df.columns):
                value = preview_df.iloc[row_index][col_name]
                self.preview_table.setItem(
                    row_index,
                    col_index,
                    QTableWidgetItem(str(value)),
                )

        self.preview_table.resizeColumnsToContents()

    def preview_data(self):

        try:
            messages = []

            matches_csv = self.matches_path.text().strip() or None
            odds_csv = self.odds_path.text().strip() or None

            if matches_csv:
                result = self.validator.load_and_validate_matches_csv(matches_csv)
                messages.append(f"Matches validation: {result['ok']}")
                messages.append(f"Matches rows: {result['row_count']}")
                if result["missing_columns"]:
                    messages.append(
                        f"Missing matches columns: {result['missing_columns']}"
                    )
                self._render_preview(result["dataframe"])

            if odds_csv and not matches_csv:
                result = self.validator.load_and_validate_odds_csv(odds_csv)
                messages.append(f"Odds validation: {result['ok']}")
                messages.append(f"Odds rows: {result['row_count']}")
                if result["missing_columns"]:
                    messages.append(
                        f"Missing odds columns: {result['missing_columns']}"
                    )
                self._render_preview(result["dataframe"])

            self.output.setPlainText(
                "\n".join(messages) if messages else "No file selected."
            )

        except Exception as exc:
            self.output.setPlainText(str(exc))
            QMessageBox.critical(self, "Preview Error", str(exc))

    def import_data(self):

        try:
            matches_csv = self.matches_path.text().strip() or None
            odds_csv = self.odds_path.text().strip() or None

            results = self.controller.import_csv_data(
                matches_csv=matches_csv,
                odds_csv=odds_csv,
            )

            self.output.setPlainText(str(results))

            QMessageBox.information(
                self,
                "Import",
                "CSV validation and import completed successfully.",
            )
        except Exception as exc:
            self.output.setPlainText(str(exc))
            QMessageBox.critical(
                self,
                "Import Error",
                str(exc),
            )

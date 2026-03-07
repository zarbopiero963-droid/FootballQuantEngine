from pathlib import Path

import pandas as pd
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from export.csv_exporter import CsvExporter
from export.excel_exporter import ExcelExporter


class DashboardCard(QFrame):

    def __init__(self, title, value="-"):

        super().__init__()

        self.setFrameShape(QFrame.Shape.StyledPanel)

        self.title_label = QLabel(title)
        self.value_label = QLabel(str(value))

        layout = QVBoxLayout()
        layout.addWidget(self.title_label)
        layout.addWidget(self.value_label)

        self.setLayout(layout)

    def set_value(self, value):

        self.value_label.setText(str(value))


class DashboardView(QWidget):

    def __init__(self):

        super().__init__()

        self.current_df = pd.DataFrame()
        self.filtered_df = pd.DataFrame()
        self.sort_ascending = True

        self.title = QLabel("Football Quant Engine Dashboard")

        self.total_bets_card = DashboardCard("Value Bets", "0")
        self.status_card = DashboardCard("Last Status", "Ready")
        self.rows_card = DashboardCard("Rows", "0")

        cards_layout = QHBoxLayout()
        cards_layout.addWidget(self.total_bets_card)
        cards_layout.addWidget(self.status_card)
        cards_layout.addWidget(self.rows_card)

        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search...")
        self.search_input.textChanged.connect(self.apply_filters)

        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.refresh_table)

        self.sort_button = QPushButton("Toggle Sort")
        self.sort_button.clicked.connect(self.toggle_sort)

        self.export_csv_button = QPushButton("Export CSV")
        self.export_csv_button.clicked.connect(self.export_csv)

        self.export_excel_button = QPushButton("Export Excel")
        self.export_excel_button.clicked.connect(self.export_excel)

        controls_layout = QHBoxLayout()
        controls_layout.addWidget(QLabel("Filter"))
        controls_layout.addWidget(self.search_input)
        controls_layout.addWidget(self.refresh_button)
        controls_layout.addWidget(self.sort_button)
        controls_layout.addWidget(self.export_csv_button)
        controls_layout.addWidget(self.export_excel_button)

        self.results_table = QTableWidget()
        self.results_table.setColumnCount(0)
        self.results_table.setRowCount(0)

        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.title)
        main_layout.addLayout(cards_layout)
        main_layout.addLayout(controls_layout)
        main_layout.addWidget(QLabel("Results"))
        main_layout.addWidget(self.results_table)
        main_layout.addWidget(QLabel("Output Log"))
        main_layout.addWidget(self.log_output)

        self.setLayout(main_layout)

    def set_status(self, text):

        self.status_card.set_value(text)

    def append_log(self, text):

        self.log_output.append(str(text))

    def set_log_text(self, text):

        self.log_output.setPlainText(str(text))

    def clear_table(self):

        self.current_df = pd.DataFrame()
        self.filtered_df = pd.DataFrame()
        self.results_table.setRowCount(0)
        self.results_table.setColumnCount(0)
        self.rows_card.set_value(0)
        self.total_bets_card.set_value(0)

    def set_results_dataframe(self, df):

        if df is None or getattr(df, "empty", False):
            self.clear_table()
            return

        self.current_df = df.copy()
        self.filtered_df = df.copy()
        self.render_dataframe(self.filtered_df)

    def set_results_from_records(self, records):

        if not records:
            self.clear_table()
            return

        df = pd.DataFrame(records)
        self.set_results_dataframe(df)

    def render_dataframe(self, df):

        if df is None or df.empty:
            self.results_table.setRowCount(0)
            self.results_table.setColumnCount(0)
            self.rows_card.set_value(0)
            self.total_bets_card.set_value(0)
            return

        columns = list(df.columns)
        rows = len(df.index)

        self.results_table.setColumnCount(len(columns))
        self.results_table.setRowCount(rows)
        self.results_table.setHorizontalHeaderLabels(columns)

        for row_index in range(rows):
            for col_index, col_name in enumerate(columns):
                value = df.iloc[row_index][col_name]
                item = QTableWidgetItem(str(value))
                self.results_table.setItem(row_index, col_index, item)

        self.rows_card.set_value(rows)
        self.total_bets_card.set_value(rows)

    def apply_filters(self):

        if self.current_df is None or self.current_df.empty:
            self.render_dataframe(pd.DataFrame())
            return

        query = self.search_input.text().strip().lower()

        if not query:
            filtered = self.current_df.copy()
        else:
            mask = self.current_df.astype(str).apply(
                lambda row: row.str.lower().str.contains(query, na=False).any(),
                axis=1,
            )
            filtered = self.current_df[mask].copy()

        self.filtered_df = filtered
        self.render_dataframe(self.filtered_df)

    def refresh_table(self):

        self.apply_filters()
        self.append_log("Dashboard refreshed.")

    def toggle_sort(self):

        if self.filtered_df is None or self.filtered_df.empty:
            return

        first_column = self.filtered_df.columns[0]
        self.filtered_df = self.filtered_df.sort_values(
            by=first_column,
            ascending=self.sort_ascending,
        )
        self.sort_ascending = not self.sort_ascending
        self.render_dataframe(self.filtered_df)
        self.append_log("Sort toggled.")

    def export_csv(self):

        if self.filtered_df is None or self.filtered_df.empty:
            self.append_log("CSV export skipped: no data.")
            return

        Path("outputs").mkdir(parents=True, exist_ok=True)
        filepath = "outputs/dashboard_export.csv"

        CsvExporter().export_value_bets(
            filepath,
            self.filtered_df.to_dict("records"),
        )
        self.append_log(f"CSV exported: {filepath}")

    def export_excel(self):

        if self.filtered_df is None or self.filtered_df.empty:
            self.append_log("Excel export skipped: no data.")
            return

        Path("outputs").mkdir(parents=True, exist_ok=True)
        filepath = "outputs/dashboard_export.xlsx"

        ExcelExporter().export_value_bets(
            filepath,
            self.filtered_df.to_dict("records"),
        )
        self.append_log(f"Excel exported: {filepath}")

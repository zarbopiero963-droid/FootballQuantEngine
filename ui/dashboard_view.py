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
    QTabWidget,
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
        self.last_csv_path = "outputs/dashboard_export.csv"
        self.last_excel_path = "outputs/dashboard_export.xlsx"

        self.title = QLabel("Football Quant Engine Dashboard")

        self.total_bets_card = DashboardCard("Value Bets", "0")
        self.status_card = DashboardCard("Last Status", "Ready")
        self.rows_card = DashboardCard("Rows", "0")
        self.columns_card = DashboardCard("Columns", "0")
        self.filter_card = DashboardCard("Filtered", "0")

        self.bankroll_card = DashboardCard("Bankroll", "100.00")
        self.roi_card = DashboardCard("ROI", "0.00")
        self.yield_card = DashboardCard("Yield", "0.00")
        self.hit_rate_card = DashboardCard("Hit Rate", "0.00")
        self.runs_card = DashboardCard("Runs", "0")
        self.max_dd_card = DashboardCard("Max Drawdown", "0.00")

        cards_row_1 = QHBoxLayout()
        cards_row_1.addWidget(self.total_bets_card)
        cards_row_1.addWidget(self.status_card)
        cards_row_1.addWidget(self.rows_card)
        cards_row_1.addWidget(self.columns_card)
        cards_row_1.addWidget(self.filter_card)

        cards_row_2 = QHBoxLayout()
        cards_row_2.addWidget(self.bankroll_card)
        cards_row_2.addWidget(self.roi_card)
        cards_row_2.addWidget(self.yield_card)
        cards_row_2.addWidget(self.hit_rate_card)
        cards_row_2.addWidget(self.runs_card)
        cards_row_2.addWidget(self.max_dd_card)

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

        self.open_csv_button = QPushButton("Open CSV")
        self.open_csv_button.clicked.connect(self.open_csv)

        self.open_excel_button = QPushButton("Open Excel")
        self.open_excel_button.clicked.connect(self.open_excel)

        controls_layout = QHBoxLayout()
        controls_layout.addWidget(QLabel("Filter"))
        controls_layout.addWidget(self.search_input)
        controls_layout.addWidget(self.refresh_button)
        controls_layout.addWidget(self.sort_button)
        controls_layout.addWidget(self.export_csv_button)
        controls_layout.addWidget(self.export_excel_button)
        controls_layout.addWidget(self.open_csv_button)
        controls_layout.addWidget(self.open_excel_button)

        self.results_table = QTableWidget()
        self.results_table.setColumnCount(0)
        self.results_table.setRowCount(0)

        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)

        self.history_output = QTextEdit()
        self.history_output.setReadOnly(True)

        self.metrics_output = QTextEdit()
        self.metrics_output.setReadOnly(True)

        self.charts_output = QTextEdit()
        self.charts_output.setReadOnly(True)

        self.inline_status = QLabel("Dashboard ready.")

        self.tabs = QTabWidget()

        results_tab = QWidget()
        results_layout = QVBoxLayout()
        results_layout.addWidget(QLabel("Results"))
        results_layout.addWidget(self.results_table)
        results_tab.setLayout(results_layout)

        log_tab = QWidget()
        log_layout = QVBoxLayout()
        log_layout.addWidget(QLabel("Output Log"))
        log_layout.addWidget(self.log_output)
        log_tab.setLayout(log_layout)

        history_tab = QWidget()
        history_layout = QVBoxLayout()
        history_layout.addWidget(QLabel("Run History"))
        history_layout.addWidget(self.history_output)
        history_tab.setLayout(history_layout)

        metrics_tab = QWidget()
        metrics_layout = QVBoxLayout()
        metrics_layout.addWidget(QLabel("Metrics Panel"))
        metrics_layout.addWidget(self.metrics_output)
        metrics_tab.setLayout(metrics_layout)

        charts_tab = QWidget()
        charts_layout = QVBoxLayout()
        charts_layout.addWidget(QLabel("Charts / Equity"))
        charts_layout.addWidget(self.charts_output)
        charts_tab.setLayout(charts_layout)

        self.tabs.addTab(results_tab, "Results")
        self.tabs.addTab(log_tab, "Log")
        self.tabs.addTab(history_tab, "History")
        self.tabs.addTab(metrics_tab, "Metrics")
        self.tabs.addTab(charts_tab, "Charts")

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.title)
        main_layout.addLayout(cards_row_1)
        main_layout.addLayout(cards_row_2)
        main_layout.addLayout(controls_layout)
        main_layout.addWidget(self.tabs)
        main_layout.addWidget(self.inline_status)

        self.setLayout(main_layout)

    def set_status(self, text):

        self.status_card.set_value(text)
        self.inline_status.setText(f"Status: {text}")

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
        self.columns_card.set_value(0)
        self.filter_card.set_value(0)
        self.inline_status.setText("Dashboard cleared.")

    def set_metrics(self, metrics):

        if not metrics:
            self.metrics_output.setPlainText("No metrics available.")
            return

        self.bankroll_card.set_value(
            f"{(metrics.get('bankroll_history') or [100])[-1]:.2f}"
        )
        self.roi_card.set_value(f"{metrics.get('roi', 0):.2%}")
        self.yield_card.set_value(f"{metrics.get('yield', 0):.2%}")
        self.hit_rate_card.set_value(f"{metrics.get('hit_rate', 0):.2%}")
        self.runs_card.set_value(metrics.get("total_bets", 0))
        self.max_dd_card.set_value(f"{metrics.get('max_drawdown', 0):.2%}")

        text = []
        text.append(f"Bankroll: {(metrics.get('bankroll_history') or [100])[-1]:.2f}")
        text.append(f"Total Profit: {metrics.get('total_profit', 0):.2f}")
        text.append(f"Total Staked: {metrics.get('total_staked', 0):.2f}")
        text.append(f"ROI: {metrics.get('roi', 0):.2%}")
        text.append(f"Yield: {metrics.get('yield', 0):.2%}")
        text.append(f"Hit Rate: {metrics.get('hit_rate', 0):.2%}")
        text.append(f"Total Bets: {metrics.get('total_bets', 0)}")
        text.append(f"Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
        text.append(f"Brier Score: {metrics.get('brier_score', 0):.4f}")
        text.append(f"Log Loss: {metrics.get('log_loss', 0):.4f}")

        self.metrics_output.setPlainText("\n".join(text))

        chart_lines = []
        chart_lines.append("Bankroll History:")
        chart_lines.append(str(metrics.get("bankroll_history", [])))
        chart_lines.append("")
        chart_lines.append("Accuracy History:")
        chart_lines.append(str(metrics.get("accuracy_history", [])))
        chart_lines.append("")
        chart_lines.append("Drawdown History:")
        chart_lines.append(str(metrics.get("drawdown_history", [])))

        self.charts_output.setPlainText("\n".join(chart_lines))

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
            self.columns_card.set_value(0)
            self.filter_card.set_value(0)
            self.inline_status.setText("No rows to display.")
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

        self.results_table.resizeColumnsToContents()
        self.results_table.resizeRowsToContents()

        self.rows_card.set_value(rows)
        self.total_bets_card.set_value(rows)
        self.columns_card.set_value(len(columns))
        self.filter_card.set_value(rows)
        self.inline_status.setText(f"Showing {rows} rows and {len(columns)} columns.")

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
        self.inline_status.setText("Dashboard refreshed.")

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
        self.inline_status.setText(f"Sorted by {first_column}.")

    def export_csv(self):

        if self.filtered_df is None or self.filtered_df.empty:
            self.append_log("CSV export skipped: no data.")
            self.inline_status.setText("CSV export skipped.")
            return

        Path("outputs").mkdir(parents=True, exist_ok=True)

        CsvExporter().export_value_bets(
            self.last_csv_path,
            self.filtered_df.to_dict("records"),
        )
        self.append_log(f"CSV exported: {self.last_csv_path}")
        self.inline_status.setText(f"CSV exported: {self.last_csv_path}")

    def export_excel(self):

        if self.filtered_df is None or self.filtered_df.empty:
            self.append_log("Excel export skipped: no data.")
            self.inline_status.setText("Excel export skipped.")
            return

        Path("outputs").mkdir(parents=True, exist_ok=True)

        ExcelExporter().export_value_bets(
            self.last_excel_path,
            self.filtered_df.to_dict("records"),
        )
        self.append_log(f"Excel exported: {self.last_excel_path}")
        self.inline_status.setText(f"Excel exported: {self.last_excel_path}")

    def open_csv(self):

        path = Path(self.last_csv_path)
        if not path.exists():
            self.append_log("Open CSV skipped: file not found.")
            self.inline_status.setText("CSV file not found.")
            return

        import webbrowser

        webbrowser.open(f"file://{path.resolve()}")
        self.append_log(f"CSV opened: {path}")
        self.inline_status.setText(f"CSV opened: {path}")

    def open_excel(self):

        path = Path(self.last_excel_path)
        if not path.exists():
            self.append_log("Open Excel skipped: file not found.")
            self.inline_status.setText("Excel file not found.")
            return

        import webbrowser

        webbrowser.open(f"file://{path.resolve()}")
        self.append_log(f"Excel opened: {path}")
        self.inline_status.setText(f"Excel opened: {path}")

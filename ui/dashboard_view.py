from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


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

        self.title = QLabel("Football Quant Engine Dashboard")

        self.total_bets_card = DashboardCard("Value Bets", "0")
        self.status_card = DashboardCard("Last Status", "Ready")
        self.rows_card = DashboardCard("Rows", "0")

        cards_layout = QHBoxLayout()
        cards_layout.addWidget(self.total_bets_card)
        cards_layout.addWidget(self.status_card)
        cards_layout.addWidget(self.rows_card)

        self.results_table = QTableWidget()
        self.results_table.setColumnCount(0)
        self.results_table.setRowCount(0)

        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.title)
        main_layout.addLayout(cards_layout)
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

        self.results_table.setRowCount(0)
        self.results_table.setColumnCount(0)
        self.rows_card.set_value(0)
        self.total_bets_card.set_value(0)

    def set_results_dataframe(self, df):

        if df is None:
            self.clear_table()
            return

        if getattr(df, "empty", False):
            self.clear_table()
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

    def set_results_from_records(self, records):

        if not records:
            self.clear_table()
            return

        columns = list(records[0].keys())
        rows = len(records)

        self.results_table.setColumnCount(len(columns))
        self.results_table.setRowCount(rows)
        self.results_table.setHorizontalHeaderLabels(columns)

        for row_index, row_data in enumerate(records):
            for col_index, col_name in enumerate(columns):
                value = row_data.get(col_name, "")
                item = QTableWidgetItem(str(value))
                self.results_table.setItem(row_index, col_index, item)

        self.rows_card.set_value(rows)
        self.total_bets_card.set_value(rows)

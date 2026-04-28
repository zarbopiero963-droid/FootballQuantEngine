from __future__ import annotations

import os

from export.csv_exporter import CsvExporter
from export.excel_exporter import ExcelExporter


class OutputManager:

    def __init__(self):

        self.csv_exporter = CsvExporter()
        self.excel_exporter = ExcelExporter()

        os.makedirs("outputs", exist_ok=True)

    def export_value_bets(self, value_bets):

        if not value_bets:
            return

        self.csv_exporter.export_value_bets(
            "outputs/value_bets.csv",
            value_bets,
        )

        self.excel_exporter.export_value_bets(
            "outputs/value_bets.xlsx",
            value_bets,
        )

    def export_dataframe(self, dataframe, name):

        if dataframe.empty:
            return

        csv_path = f"outputs/{name}.csv"
        xlsx_path = f"outputs/{name}.xlsx"

        dataframe.to_csv(csv_path, index=False)
        dataframe.to_excel(xlsx_path, index=False)

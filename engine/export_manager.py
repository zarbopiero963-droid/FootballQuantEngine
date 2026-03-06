from export.csv_exporter import CsvExporter
from export.excel_exporter import ExcelExporter


class ExportManager:

    def __init__(self):

        self.csv_exporter = CsvExporter()
        self.excel_exporter = ExcelExporter()

    def export(self, value_bets):

        if not value_bets:
            return

        self.csv_exporter.export_value_bets(
            "value_bets.csv",
            value_bets,
        )

        self.excel_exporter.export_value_bets(
            "value_bets.xlsx",
            value_bets,
        )

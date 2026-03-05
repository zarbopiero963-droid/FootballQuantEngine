import pandas as pd


class ExcelExporter:

    def export_value_bets(self, filepath, value_bets):

        df = pd.DataFrame(value_bets)
        df.to_excel(filepath, index=False)

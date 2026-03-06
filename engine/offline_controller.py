from engine.offline_engine import OfflineEngine


class OfflineController:

    def __init__(self):

        self.engine = OfflineEngine()

    def import_csv_data(self, matches_csv=None, odds_csv=None):

        self.engine.import_csv_data(
            matches_csv=matches_csv,
            odds_csv=odds_csv,
        )

    def run_reports(self, ranked_df=None):

        return self.engine.run_reports(ranked_df=ranked_df)

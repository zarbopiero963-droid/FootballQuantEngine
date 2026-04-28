from engine.offline_engine import OfflineEngine


class OfflineController:

    def __init__(self):

        self.engine = OfflineEngine()

    def import_csv_data(self, matches_csv=None, odds_csv=None):

        results = {}

        if matches_csv:
            results["matches"] = self.engine.csv_importer.import_matches(matches_csv)

        if odds_csv:
            results["odds"] = self.engine.csv_importer.import_odds(odds_csv)

        return results

    def run_reports(self):

        return self.engine.run_reports()

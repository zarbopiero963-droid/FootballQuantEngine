import pandas as pd

from analysis.manual_context import ManualContextManager
from analysis.probability_markets import ProbabilityMarkets
from analysis.report_generator import ReportGenerator
from features.offline_features import OfflineFeatureBuilder
from models.bayesian_model import BayesianModel
from notifications.desktop_popup import DesktopPopup
from training.backtest_engine import BacktestEngine
from training.backtest_metrics import BacktestMetrics
from training.csv_importer import CsvImporter
from training.dataset_builder import DatasetBuilder
from training.local_automl import LocalAutoML


class OfflineEngine:

    def __init__(self):

        self.csv_importer = CsvImporter()
        self.feature_builder = OfflineFeatureBuilder()
        self.bayesian = BayesianModel()
        self.markets = ProbabilityMarkets()
        self.manual_context = ManualContextManager()
        self.dataset_builder = DatasetBuilder()
        self.backtest_engine = BacktestEngine()
        self.backtest_metrics = BacktestMetrics()
        self.report_generator = ReportGenerator()
        self.desktop_popup = DesktopPopup()
        self.local_automl = LocalAutoML()

    def import_csv_data(self, matches_csv=None, odds_csv=None):

        if matches_csv:
            self.csv_importer.import_matches(matches_csv)

        if odds_csv:
            self.csv_importer.import_odds(odds_csv)

    def run_offline_analysis(self, fixtures_df):

        if fixtures_df.empty:
            return {
                "features": pd.DataFrame(),
                "predictions": pd.DataFrame(),
            }

        features_df = self.feature_builder.build_team_offline_features(fixtures_df)

        rows = []

        for _, row in features_df.iterrows():

            bayes = self.bayesian.predict(
                home_attack=row["goals_for_avg"],
                home_defense=row["goals_against_avg"],
                away_attack=row["goals_for_avg"],
                away_defense=row["goals_against_avg"],
            )

            goal_rate = max(row["goals_for_avg"], 0.1)
            concede_rate = max(row["goals_against_avg"], 0.1)

            over_under = self.markets.over_under_25(
                lambda_home=goal_rate,
                lambda_away=concede_rate,
            )

            btts = self.markets.btts(
                lambda_home=goal_rate,
                lambda_away=concede_rate,
            )

            rows.append(
                {
                    "team": row["team"],
                    "home_win": bayes["home_win"],
                    "draw": bayes["draw"],
                    "away_win": bayes["away_win"],
                    "over_2_5": over_under["over_2_5"],
                    "under_2_5": over_under["under_2_5"],
                    "btts_yes": btts["btts_yes"],
                    "btts_no": btts["btts_no"],
                    "clean_sheet_rate": row["clean_sheet_rate"],
                    "over_2_5_rate": row["over_2_5_rate"],
                    "btts_rate": row["btts_rate"],
                    "rolling_goals_for_avg": row["rolling_goals_for_avg"],
                    "rolling_goals_against_avg": row["rolling_goals_against_avg"],
                    "weighted_form_score": row["weighted_form_score"],
                }
            )

        return {
            "features": features_df,
            "predictions": pd.DataFrame(rows),
        }

    def run_reports(self, ranked_df=None):

        backtest_df = self.backtest_engine.run()
        metrics = self.backtest_metrics.calculate(backtest_df)

        self.report_generator.generate(
            metrics=metrics,
            ranked_df=ranked_df,
        )

        self.desktop_popup.notify(
            "Football Quant Engine",
            "Advanced report generated in outputs/advanced_report.html",
        )

        dataset_df = self.dataset_builder.build_training_dataset()
        automl_status = self.local_automl.run(dataset_df)

        return {
            "metrics": metrics,
            "automl": automl_status,
        }

import pandas as pd

from config.settings_manager import load_settings
from data.api_football_collector import ApiFootballCollector
from data.odds_collector import OddsCollector
from database.db_manager import init_db
from engine.fixture_mapper import map_api_football_response_to_fixtures
from engine.match_ranker import MatchRanker
from engine.odds_mapper import map_odds_events_to_match_id
from engine.output_manager import OutputManager
from engine.prediction_pipeline import PredictionPipeline
from engine.repository_saver import RepositorySaver
from engine.training_manager import TrainingManager
from engine.value_bet_notifier import ValueBetNotifier
from features.feature_engine import FeatureEngine


class JobRunner:

    def __init__(self):

        self.api = ApiFootballCollector()
        self.odds = OddsCollector()
        self.pipeline = PredictionPipeline()
        self.saver = RepositorySaver()
        self.notifier = ValueBetNotifier()
        self.ranker = MatchRanker()
        self.output_manager = OutputManager()
        self.training_manager = TrainingManager()

        init_db()

    def run_cycle(self):

        settings = load_settings()

        if not settings.api_football_key or not settings.odds_api_key:
            raise Exception("Missing API keys in settings.json")

        fixtures_response = self.api.get_next_matches()
        fixtures = map_api_football_response_to_fixtures(fixtures_response)

        rows = []

        for fixture in fixtures:
            rows.append(
                {
                    "home": fixture["home"],
                    "away": fixture["away"],
                    "home_goals": fixture.get("home_goals") or 0,
                    "away_goals": fixture.get("away_goals") or 0,
                }
            )

        fixtures_df = pd.DataFrame(rows)

        feature_engine = FeatureEngine()
        features_df = feature_engine.build_features(fixtures_df)

        odds_events = self.odds.get_odds()
        odds_data = map_odds_events_to_match_id(odds_events)

        value_bets = self.pipeline.run(features_df, odds_data)

        self.saver.save_all(fixtures, odds_data, value_bets)
        self.notifier.notify(value_bets)

        ranked = self.ranker.rank(value_bets)
        self.output_manager.export_value_bets(value_bets)

        if not ranked.empty:
            self.output_manager.export_dataframe(
                ranked,
                "ranked_value_bets",
            )

        dataset_df = self.training_manager.build_dataset()

        if not dataset_df.empty:
            self.output_manager.export_dataframe(
                dataset_df,
                "training_dataset",
            )

        backtest_df = self.training_manager.run_backtest()

        if not backtest_df.empty:
            self.output_manager.export_dataframe(
                backtest_df,
                "backtest_results",
            )

        return ranked

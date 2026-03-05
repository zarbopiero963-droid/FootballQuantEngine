from data.api_football_collector import ApiFootballCollector
from data.odds_collector import OddsCollector
from engine.prediction_pipeline import PredictionPipeline
from features.feature_engine import FeatureEngine


class JobRunner:

    def __init__(self):

        self.api = ApiFootballCollector()
        self.odds = OddsCollector()
        self.pipeline = PredictionPipeline()

    def run_cycle(self):

        fixtures_response = self.api.get_next_matches()

        fixtures = []

        for match in fixtures_response.get("response", []):

            fixture = match.get("fixture", {})
            teams = match.get("teams", {})

            fixtures.append(
                {
                    "home": teams.get("home", {}).get("name"),
                    "away": teams.get("away", {}).get("name"),
                    "home_goals": 0,
                    "away_goals": 0,
                }
            )

        import pandas as pd

        fixtures_df = pd.DataFrame(fixtures)

        feature_engine = FeatureEngine()

        features_df = feature_engine.build_features(fixtures_df)

        odds_response = self.odds.get_odds()

        odds_data = {}

        for event in odds_response:

            home = event.get("home_team")
            away = event.get("away_team")

            match_id = f"{home}_vs_{away}"

            bookmakers = event.get("bookmakers", [])

            if not bookmakers:
                continue

            markets = bookmakers[0].get("markets", [])

            if not markets:
                continue

            outcomes = markets[0].get("outcomes", [])

            odds_map = {}

            for outcome in outcomes:

                name = outcome.get("name")
                price = outcome.get("price")

                if name == home:
                    odds_map["home"] = price
                elif name == away:
                    odds_map["away"] = price
                elif name.lower() == "draw":
                    odds_map["draw"] = price

            odds_data[match_id] = odds_map

        value_bets = self.pipeline.run(features_df, odds_data)

        return value_bets

from database.fixtures_repository import FixturesRepository
from database.odds_repository import OddsRepository
from database.prediction_repository import PredictionRepository
from strategies.edge_detector import EdgeDetector


class RepositorySaver:

    def __init__(self):

        self.fixtures_repo = FixturesRepository()
        self.odds_repo = OddsRepository()
        self.pred_repo = PredictionRepository()
        self.edge_detector = EdgeDetector()

    def save_all(self, fixtures, odds_data, value_bets):

        for f in fixtures:
            self.fixtures_repo.save_fixture(f)

        for match_id, odds in odds_data.items():

            fixture_id = None
            for f in fixtures:
                if f"{f['home']}_vs_{f['away']}" == match_id:
                    fixture_id = f["fixture_id"]
                    break

            if fixture_id is None:
                continue

            self.odds_repo.save_odds(
                fixture_id, odds.get("home"), odds.get("draw"), odds.get("away")
            )

        for vb in value_bets:

            match_id = vb.get("match_id")

            fixture_id = None
            for f in fixtures:
                if f"{f['home']}_vs_{f['away']}" == match_id:
                    fixture_id = f["fixture_id"]
                    break

            if fixture_id is None:
                continue

            prob = vb.get("probability")
            odds = vb.get("odds")

            edge = self.edge_detector.calculate_edge(prob, odds)

            if vb.get("market") == "home":
                home_prob = prob
                draw_prob = None
                away_prob = None
            elif vb.get("market") == "draw":
                home_prob = None
                draw_prob = prob
                away_prob = None
            else:
                home_prob = None
                draw_prob = None
                away_prob = prob

            self.pred_repo.save_prediction(
                fixture_id=fixture_id,
                model="ensemble",
                home_prob=home_prob,
                draw_prob=draw_prob,
                away_prob=away_prob,
                edge=edge,
            )

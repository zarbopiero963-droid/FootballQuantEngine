from models.poisson_model import PoissonModel
from simulation.monte_carlo import MonteCarloSimulator
from strategies.edge_detector import EdgeDetector
from strategies.value_bet_strategy import ValueBetStrategy


class PredictionPipeline:

    def __init__(self):

        self.poisson = PoissonModel()
        self.simulator = MonteCarloSimulator()
        self.edge_detector = EdgeDetector()
        self.strategy = ValueBetStrategy()

    def run(self, features_df, odds_data):

        predictions = {}

        for _, row in features_df.iterrows():

            match_id = f"{row['home']}_vs_{row['away']}"

            model_probs = self.poisson.predict(
                row["home_attack"],
                row["home_defense"],
                row["away_attack"],
                row["away_defense"],
            )

            lambda_home = row["home_attack"] * row["away_defense"]
            lambda_away = row["away_attack"] * row["home_defense"]

            simulation_probs = self.simulator.simulate(lambda_home, lambda_away)

            predictions[match_id] = {
                "home_win": (model_probs["home_win"] + simulation_probs["home_win"])
                / 2,
                "draw": (model_probs["draw"] + simulation_probs["draw"]) / 2,
                "away_win": (model_probs["away_win"] + simulation_probs["away_win"])
                / 2,
            }

        value_bets = self.strategy.find_value_bets(
            predictions, odds_data, self.edge_detector
        )

        return value_bets

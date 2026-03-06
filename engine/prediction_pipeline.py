from engine.plugin_loader import PluginLoader
from simulation.monte_carlo import MonteCarloSimulator
from strategies.edge_detector import EdgeDetector
from strategies.value_bet_strategy import ValueBetStrategy


class PredictionPipeline:

    def __init__(self):

        self.loader = PluginLoader()
        self.simulator = MonteCarloSimulator()
        self.edge_detector = EdgeDetector()
        self.strategy = ValueBetStrategy()

    def run(self, features_df, odds_data):

        plugins = self.loader.load_plugins()

        predictions = {}

        for _, row in features_df.iterrows():

            match_id = f"{row['home']}_vs_{row['away']}"

            plugin_outputs = []

            for plugin in plugins:
                plugin_outputs.append(plugin.predict(row))

            if not plugin_outputs:
                continue

            home_prob = sum(output["home_win"] for output in plugin_outputs) / len(
                plugin_outputs
            )

            draw_prob = sum(output["draw"] for output in plugin_outputs) / len(
                plugin_outputs
            )

            away_prob = sum(output["away_win"] for output in plugin_outputs) / len(
                plugin_outputs
            )

            lambda_home = row["home_attack"] * row["away_defense"]
            lambda_away = row["away_attack"] * row["home_defense"]

            simulation_probs = self.simulator.simulate(
                lambda_home,
                lambda_away,
            )

            predictions[match_id] = {
                "home_win": (home_prob + simulation_probs["home_win"]) / 2,
                "draw": (draw_prob + simulation_probs["draw"]) / 2,
                "away_win": (away_prob + simulation_probs["away_win"]) / 2,
            }

        value_bets = self.strategy.find_value_bets(
            predictions,
            odds_data,
            self.edge_detector,
        )

        return value_bets

from engine.confidence_engine import ConfidenceEngine
from engine.market_comparison import MarketComparison
from engine.model_agreement import ModelAgreement
from engine.plugin_loader import PluginLoader
from engine.probability_calibrator import ProbabilityCalibrator
from simulation.monte_carlo import MonteCarloSimulator
from strategies.edge_detector import EdgeDetector
from strategies.no_bet_filter import NoBetFilter


class PredictionPipeline:

    def __init__(self):

        self.loader = PluginLoader()
        self.simulator = MonteCarloSimulator()
        self.edge_detector = EdgeDetector()
        self.calibrator = ProbabilityCalibrator()
        self.agreement_engine = ModelAgreement()
        self.market = MarketComparison()
        self.confidence_engine = ConfidenceEngine()
        self.no_bet_filter = NoBetFilter()

    def _combine_probabilities(self, plugin_outputs, simulation_probs):

        home_prob = sum(output["home_win"] for output in plugin_outputs) / len(
            plugin_outputs
        )

        draw_prob = sum(output["draw"] for output in plugin_outputs) / len(
            plugin_outputs
        )

        away_prob = sum(output["away_win"] for output in plugin_outputs) / len(
            plugin_outputs
        )

        combined = {
            "home_win": (home_prob + simulation_probs["home_win"]) / 2,
            "draw": (draw_prob + simulation_probs["draw"]) / 2,
            "away_win": (away_prob + simulation_probs["away_win"]) / 2,
        }

        return self.calibrator.calibrate(combined)

    def run(self, features_df, odds_data):

        plugins = self.loader.load_plugins()

        results = []

        for _, row in features_df.iterrows():

            match_id = f"{row['home']}_vs_{row['away']}"

            odds = odds_data.get(match_id)
            if not odds:
                continue

            plugin_outputs = []

            for plugin in plugins:
                plugin_outputs.append(plugin.predict(row))

            if not plugin_outputs:
                continue

            lambda_home = row["home_attack"] * row["away_defense"]
            lambda_away = row["away_attack"] * row["home_defense"]

            simulation_probs = self.simulator.simulate(
                lambda_home,
                lambda_away,
            )

            final_probs = self._combine_probabilities(
                plugin_outputs,
                simulation_probs,
            )

            agreement = self.agreement_engine.calculate(plugin_outputs)
            market_probs = self.market.implied_probabilities(odds)
            edge_map = self.market.edge_vs_market(final_probs, market_probs)

            market_labels = {
                "home": "home_win",
                "draw": "draw",
                "away": "away_win",
            }

            odds_labels = {
                "home": odds.get("home"),
                "draw": odds.get("draw"),
                "away": odds.get("away"),
            }

            for market_name, prob_key in market_labels.items():

                probability = float(final_probs.get(prob_key, 0.0))
                edge_vs_market = float(edge_map.get(prob_key, 0.0))
                odds_value = float(odds_labels.get(market_name) or 0.0)

                if odds_value <= 0:
                    continue

                model_edge = self.edge_detector.calculate_edge(
                    probability,
                    odds_value,
                )

                confidence = self.confidence_engine.score(
                    probability,
                    edge_vs_market,
                    agreement,
                )

                decision = self.no_bet_filter.decide(
                    probability=probability,
                    edge=edge_vs_market,
                    confidence=confidence,
                    agreement=agreement,
                )

                results.append(
                    {
                        "match_id": match_id,
                        "market": market_name,
                        "probability": probability,
                        "odds": odds_value,
                        "model_edge": model_edge,
                        "market_edge": edge_vs_market,
                        "confidence": confidence,
                        "agreement": agreement,
                        "decision": decision,
                    }
                )

        results.sort(
            key=lambda x: (
                x["decision"] != "BET",
                -x["confidence"],
                -x["market_edge"],
            )
        )

        return results

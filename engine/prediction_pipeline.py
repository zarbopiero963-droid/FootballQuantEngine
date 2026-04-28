from __future__ import annotations

from engine.confidence_engine import ConfidenceEngine
from engine.market_comparison import MarketComparison
from engine.model_agreement import ModelAgreement
from engine.plugin_loader import PluginLoader
from engine.probability_calibrator import ProbabilityCalibrator
from models.bivariate_poisson_model import BivariatePoissonModel
from models.xg_model import XGModel
from simulation.monte_carlo_advanced import MonteCarloAdvanced
from strategies.edge_detector import EdgeDetector
from strategies.no_bet_filter import NoBetFilter

# Weights for ensemble combination
_WEIGHTS = {
    "plugins": 0.35,  # plugin model average (Poisson, ELO, …)
    "xg": 0.25,  # xG model
    "bivariate": 0.20,  # Bivariate Poisson
    "monte_carlo": 0.20,  # Advanced Monte Carlo
}


class PredictionPipeline:
    """
    Multi-model prediction pipeline.

    Model ensemble:
      1. Plugin models      (Poisson, ELO, Bayesian via plugin loader)
      2. XGModel            (xG-based Poisson with DC correction)
      3. BivariatePoissonModel (Karlis-Ntzoufras with shared goals)
      4. MonteCarloAdvanced  (100k simulations with antithetic variates)

    All 1×2 outputs are weighted and calibrated before edge calculation.
    O/U and BTTS probabilities come from the MC simulation.
    """

    def __init__(
        self,
        completed_matches: list[dict] | None = None,
        simulations: int = 100_000,
    ) -> None:
        self.loader = PluginLoader()
        self.xg_model = XGModel()
        self.bvp_model = BivariatePoissonModel()
        self.mc_simulator = MonteCarloAdvanced()
        self.edge_detector = EdgeDetector()
        self.calibrator = ProbabilityCalibrator()
        self.agreement_engine = ModelAgreement()
        self.market = MarketComparison()
        self.confidence_engine = ConfidenceEngine()
        self.no_bet_filter = NoBetFilter()
        self.simulations = simulations

        if completed_matches:
            self.fit(completed_matches)

    def fit(self, completed_matches: list[dict]) -> "PredictionPipeline":
        """Fit all data-driven models from completed match history."""
        self.xg_model.fit(completed_matches)
        self.bvp_model.fit(completed_matches)
        return self

    def _combine_probabilities(
        self,
        plugin_outputs: list[dict],
        xg_probs: dict,
        bvp_probs: dict,
        mc_probs: dict,
    ) -> dict:
        """Weighted ensemble of all model outputs."""
        # Plugin average
        n_plugins = len(plugin_outputs)
        if n_plugins:
            p_hw = sum(o.get("home_win", 0.0) for o in plugin_outputs) / n_plugins
            p_dr = sum(o.get("draw", 0.0) for o in plugin_outputs) / n_plugins
            p_aw = sum(o.get("away_win", 0.0) for o in plugin_outputs) / n_plugins
        else:
            p_hw = p_dr = p_aw = 1 / 3

        w = _WEIGHTS
        total_w = (
            (w["plugins"] if n_plugins else 0)
            + w["xg"]
            + w["bivariate"]
            + w["monte_carlo"]
        )

        hw = (
            (p_hw * (w["plugins"] if n_plugins else 0))
            + xg_probs.get("home_win", 1 / 3) * w["xg"]
            + bvp_probs.get("home_win", 1 / 3) * w["bivariate"]
            + mc_probs.get("home_win", 1 / 3) * w["monte_carlo"]
        ) / total_w

        dr = (
            (p_dr * (w["plugins"] if n_plugins else 0))
            + xg_probs.get("draw", 1 / 3) * w["xg"]
            + bvp_probs.get("draw", 1 / 3) * w["bivariate"]
            + mc_probs.get("draw", 1 / 3) * w["monte_carlo"]
        ) / total_w

        aw = (
            (p_aw * (w["plugins"] if n_plugins else 0))
            + xg_probs.get("away_win", 1 / 3) * w["xg"]
            + bvp_probs.get("away_win", 1 / 3) * w["bivariate"]
            + mc_probs.get("away_win", 1 / 3) * w["monte_carlo"]
        ) / total_w

        total = hw + dr + aw or 1.0
        combined = {
            "home_win": hw / total,
            "draw": dr / total,
            "away_win": aw / total,
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

            home = str(row["home"])
            away = str(row["away"])

            # --- Plugin models ---
            plugin_outputs = [plugin.predict(row) for plugin in plugins]

            # --- xG model ---
            try:
                xg_probs = self.xg_model.predict(row)
            except Exception:
                xg_probs = {"home_win": 1 / 3, "draw": 1 / 3, "away_win": 1 / 3}

            # --- Bivariate Poisson ---
            try:
                bvp_out = self.bvp_model.predict_teams(home, away)
                bvp_probs = {
                    "home_win": bvp_out.get("home_win", 1 / 3),
                    "draw": bvp_out.get("draw", 1 / 3),
                    "away_win": bvp_out.get("away_win", 1 / 3),
                }
            except Exception:
                bvp_probs = {"home_win": 1 / 3, "draw": 1 / 3, "away_win": 1 / 3}

            # --- Advanced Monte Carlo ---
            lambda_home = float(row.get("home_attack", 1.2)) * float(
                row.get("away_defense", 1.0)
            )
            lambda_away = float(row.get("away_attack", 1.0)) * float(
                row.get("home_defense", 1.2)
            )
            try:
                mc_result = self.mc_simulator.simulate(
                    lambda_home,
                    lambda_away,
                    simulations=self.simulations,
                )
                mc_probs = {
                    "home_win": mc_result.home_win,
                    "draw": mc_result.draw,
                    "away_win": mc_result.away_win,
                    "over_25": mc_result.over_25,
                    "under_25": mc_result.under_25,
                    "btts_yes": mc_result.btts_yes,
                    "btts_no": mc_result.btts_no,
                }
            except Exception:
                mc_probs = {"home_win": 1 / 3, "draw": 1 / 3, "away_win": 1 / 3}

            final_probs = self._combine_probabilities(
                plugin_outputs,
                xg_probs,
                bvp_probs,
                mc_probs,
            )

            # Propagate MC side-markets into final output
            final_probs["over_25"] = mc_probs.get("over_25", 0.5)
            final_probs["under_25"] = mc_probs.get("under_25", 0.5)
            final_probs["btts_yes"] = mc_probs.get("btts_yes", 0.5)
            final_probs["btts_no"] = mc_probs.get("btts_no", 0.5)

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

                model_edge = self.edge_detector.calculate_edge(probability, odds_value)

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
                        # Extra MC markets for display
                        "over_25": final_probs.get("over_25"),
                        "btts_yes": final_probs.get("btts_yes"),
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

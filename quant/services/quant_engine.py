from quant.features.dataset_builder import QuantDatasetBuilder
from quant.features.feature_builder import FeatureBuilder
from quant.models.calibration import ProbabilityCalibration
from quant.models.elo_engine import EloEngine
from quant.models.form_engine import FormEngine
from quant.models.manual_blend_model import ManualBlendModel
from quant.models.poisson_engine import PoissonEngine
from quant.models.prediction_result import PredictionResult
from quant.services.agreement_engine import AgreementEngine
from quant.services.confidence_engine import QuantConfidenceEngine
from quant.services.market_tools import MarketTools
from quant.services.no_bet_filter import QuantNoBetFilter
from quant.services.ranker import QuantRanker


class QuantEngine:

    def __init__(self, fixtures_provider, odds_provider, advanced_provider):
        self.dataset_builder = QuantDatasetBuilder(
            fixtures_provider=fixtures_provider,
            odds_provider=odds_provider,
            advanced_provider=advanced_provider,
        )
        self.elo_engine = EloEngine()
        self.poisson_engine = PoissonEngine()
        self.form_engine = FormEngine()
        self.blend_model = ManualBlendModel()
        self.calibration = ProbabilityCalibration()
        self.market_tools = MarketTools()
        self.agreement_engine = AgreementEngine()
        self.confidence_engine = QuantConfidenceEngine()
        self.no_bet_filter = QuantNoBetFilter()
        self.ranker = QuantRanker()

    def fit(self, league=None, season=None):
        training = self.dataset_builder.build_training_data(
            league=league,
            season=season,
        )

        completed_matches = training["completed_matches"]
        advanced_stats = training["advanced_stats"]

        self.elo_engine.fit(completed_matches)
        self.poisson_engine.fit(completed_matches)
        self.form_engine.fit(completed_matches)

        self.feature_builder = FeatureBuilder(
            elo_engine=self.elo_engine,
            poisson_engine=self.poisson_engine,
            form_engine=self.form_engine,
            advanced_stats=advanced_stats,
        )

    def _three_way_component_bundle(self, features, bookmaker_odds):
        home_team = features["home_team"]
        away_team = features["away_team"]

        poisson_probs = self.poisson_engine.probabilities_1x2(home_team, away_team)
        market_probs = self.market_tools.normalize_implied_probs_1x2(bookmaker_odds)

        blended = self.blend_model.combine(
            poisson_probs=poisson_probs,
            elo_diff=features["elo_diff"],
            form_diff=features["form_diff"],
            xg_diff=features["xg_diff"],
            market_probs=market_probs,
        )

        calibrated = self.calibration.calibrate_three_way(blended)

        agreement = self.agreement_engine.three_way_agreement(
            [
                poisson_probs,
                self.blend_model.elo_to_probs(features["elo_diff"]),
                self.blend_model.form_to_probs(features["form_diff"]),
                self.blend_model.xg_to_probs(features["xg_diff"]),
                market_probs,
            ]
        )

        return {
            "poisson_probs": poisson_probs,
            "market_probs": market_probs,
            "final_probs": calibrated,
            "agreement": agreement,
        }

    def _build_1x2_results(self, fixture, features, bookmaker_odds):
        bundle = self._three_way_component_bundle(features, bookmaker_odds)

        final_probs = bundle["final_probs"]
        market_probs = bundle["market_probs"]
        agreement = bundle["agreement"]

        xg_support = 0.0
        if features["xg_diff"] > 0.20:
            xg_support = 1.0
        elif features["xg_diff"] < -0.20:
            xg_support = 0.6
        else:
            xg_support = 0.4

        mapping = {
            "home": ("home_win", float(bookmaker_odds.get("home", 0.0))),
            "draw": ("draw", float(bookmaker_odds.get("draw", 0.0))),
            "away": ("away_win", float(bookmaker_odds.get("away", 0.0))),
        }

        results = []

        for market_name, (prob_key, bookmaker_price) in mapping.items():
            probability = float(final_probs.get(prob_key, 0.0))
            fair_odds = self.market_tools.fair_odds(probability)
            model_edge = self.market_tools.edge(probability, bookmaker_price)
            market_edge = self.market_tools.probability_edge_vs_market(
                probability,
                market_probs.get(prob_key, 0.0),
            )
            confidence = self.confidence_engine.score(
                probability=probability,
                edge=market_edge,
                agreement=agreement,
                xg_support=xg_support,
            )
            decision = self.no_bet_filter.decide(
                probability=probability,
                edge=market_edge,
                confidence=confidence,
                agreement=agreement,
            )

            results.append(
                PredictionResult(
                    fixture_id=str(fixture["fixture_id"]),
                    home_team=fixture["home_team"],
                    away_team=fixture["away_team"],
                    market=market_name,
                    probability=round(probability, 6),
                    fair_odds=round(fair_odds, 4),
                    bookmaker_odds=round(bookmaker_price, 4),
                    market_edge=round(market_edge, 6),
                    model_edge=round(model_edge, 6),
                    confidence=round(confidence, 6),
                    agreement=round(agreement, 6),
                    decision=decision,
                    details={
                        "elo_diff": round(features["elo_diff"], 4),
                        "form_diff": round(features["form_diff"], 4),
                        "xg_diff": round(features["xg_diff"], 4),
                        "expected_goals_home": round(
                            features["expected_goals_home"], 4
                        ),
                        "expected_goals_away": round(
                            features["expected_goals_away"], 4
                        ),
                    },
                )
            )

        return results

    def predict(self, league=None, season=None):
        prediction_data = self.dataset_builder.build_prediction_data(
            league=league,
            season=season,
        )

        upcoming_matches = prediction_data["upcoming_matches"]
        odds_map = prediction_data["odds"]

        features_list = self.feature_builder.build_many(upcoming_matches)

        all_results = []

        for fixture, features in zip(upcoming_matches, features_list):
            fixture_id = fixture["fixture_id"]
            bookmaker_odds = odds_map.get(
                fixture_id,
                {
                    "home": 2.40,
                    "draw": 3.10,
                    "away": 2.95,
                },
            )

            fixture_results = self._build_1x2_results(
                fixture=fixture,
                features=features,
                bookmaker_odds=bookmaker_odds,
            )
            all_results.extend(fixture_results)

        ranked = self.ranker.rank(all_results)
        return ranked

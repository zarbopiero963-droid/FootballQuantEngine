from quant.features.dataset_builder import QuantDatasetBuilder
from quant.features.feature_builder import FeatureBuilder
from quant.models.calibration import ProbabilityCalibration
from quant.models.dixon_coles_engine import DixonColesEngine
from quant.models.elo_engine import EloEngine
from quant.models.form_engine import FormEngine
from quant.models.goal_momentum_engine import GoalMomentumEngine
from quant.models.h2h_engine import H2HEngine
from quant.models.injury_engine import InjuryEngine
from quant.models.manual_blend_model import ManualBlendModel
from quant.models.prediction_result import PredictionResult
from quant.models.referee_engine import RefereeEngine
from quant.models.rest_engine import RestEngine
from quant.models.standings_engine import StandingsEngine
from quant.services.agreement_engine import AgreementEngine
from quant.services.confidence_engine import QuantConfidenceEngine
from quant.services.market_tools import MarketTools
from quant.services.no_bet_filter import QuantNoBetFilter
from quant.services.ranker import QuantRanker


class QuantEngine:
    def __init__(self, fixtures_provider, odds_provider, weather_engine=None):
        self.dataset_builder = QuantDatasetBuilder(
            fixtures_provider=fixtures_provider,
            odds_provider=odds_provider,
        )
        # Models
        self.dc_engine = DixonColesEngine()
        self.elo_engine = EloEngine()
        self.form_engine = FormEngine()
        self.h2h_engine = H2HEngine()
        self.momentum_engine = GoalMomentumEngine()
        self.rest_engine = RestEngine()
        self.standings_engine = StandingsEngine()
        self.injury_engine = InjuryEngine()
        self.referee_engine = RefereeEngine()
        self.weather_engine = weather_engine

        # Scoring utilities
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
        completed = training["completed_matches"]

        # Fit all signal engines from historical matches
        self.dc_engine.fit(completed)
        self.elo_engine.fit(completed)
        self.form_engine.fit(completed)
        self.h2h_engine.fit(completed)
        self.momentum_engine.fit(completed)
        self.rest_engine.fit(completed)

        # Fit context engines from enriched training data
        self.standings_engine.fit(training.get("standings", []))
        self.injury_engine.fit(training.get("injuries", {}))
        self.referee_engine.fit(training.get("referee_stats", {}))

        self.feature_builder = FeatureBuilder(
            elo_engine=self.elo_engine,
            dc_engine=self.dc_engine,
            form_engine=self.form_engine,
            h2h_engine=self.h2h_engine,
            momentum_engine=self.momentum_engine,
            rest_engine=self.rest_engine,
            standings_engine=self.standings_engine,
            injury_engine=self.injury_engine,
            referee_engine=self.referee_engine,
            weather_engine=self.weather_engine,
            advanced_stats=training.get("advanced_stats", {}),
        )

    def _three_way_component_bundle(self, features, bookmaker_odds):
        dc_probs = features["dc_probs"]
        market_probs = self.market_tools.normalize_implied_probs_1x2(bookmaker_odds)

        blended = self.blend_model.combine(
            dc_probs=dc_probs,
            elo_diff=features["elo_diff"],
            form_diff=features["form_diff"],
            xg_diff=features.get("xg_diff", 0.0),
            market_probs=market_probs,
            h2h_diff=features.get("h2h_diff", 0.0),
            momentum_diff=features.get("momentum_diff", 0.0),
            motivation_diff=features.get("motivation_diff", 0.0),
        )

        calibrated = self.calibration.calibrate_three_way(blended)

        agreement = self.agreement_engine.three_way_agreement(
            [
                dc_probs,
                self.blend_model.elo_to_probs(features["elo_diff"]),
                self.blend_model.form_to_probs(features["form_diff"]),
                self.blend_model.xg_to_probs(features.get("xg_diff", 0.0)),
                self.blend_model.h2h_to_probs(features.get("h2h_diff", 0.0)),
                market_probs,
            ]
        )

        return {
            "dc_probs": dc_probs,
            "market_probs": market_probs,
            "final_probs": calibrated,
            "agreement": agreement,
        }

    def _build_1x2_results(self, fixture, features, bookmaker_odds):
        bundle = self._three_way_component_bundle(features, bookmaker_odds)

        final_probs = bundle["final_probs"]
        market_probs = bundle["market_probs"]
        agreement = bundle["agreement"]

        xg_diff = features.get("xg_diff", 0.0)
        if xg_diff > 0.20:
            xg_support = 1.0
        elif xg_diff < -0.20:
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
                probability, market_probs.get(prob_key, 0.0)
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
                        "xg_diff": round(features.get("xg_diff", 0.0), 4),
                        "h2h_diff": round(features.get("h2h_diff", 0.0), 4),
                        "momentum_diff": round(features.get("momentum_diff", 0.0), 4),
                        "motivation_diff": round(
                            features.get("motivation_diff", 0.0), 4
                        ),
                        "rest_diff": round(features.get("rest_diff", 0.0), 4),
                        "injury_diff": round(features.get("injury_diff", 0.0), 4),
                        "expected_goals_home": round(
                            features["expected_goals_home"], 4
                        ),
                        "expected_goals_away": round(
                            features["expected_goals_away"], 4
                        ),
                        "dc_rho": round(self.dc_engine.rho, 4),
                        "referee": features.get("referee", ""),
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
                {"home": 2.40, "draw": 3.10, "away": 2.95},
            )
            fixture_results = self._build_1x2_results(
                fixture=fixture,
                features=features,
                bookmaker_odds=bookmaker_odds,
            )
            all_results.extend(fixture_results)

        return self.ranker.rank(all_results)

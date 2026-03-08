from __future__ import annotations

from quant.value.clv_estimator import CLVEstimator
from quant.value.ev_calculator import EVCalculator
from quant.value.stake_policy import StakePolicy
from quant.value.value_bet_filter import ValueBetFilter
from quant.value.value_bet_ranker import ValueBetRanker


class MarketValueEngine:

    def __init__(self):
        self.ev = EVCalculator()
        self.clv = CLVEstimator()
        self.stake_policy = StakePolicy()
        self.filter = ValueBetFilter(
            min_ev=0.015,
            min_probability=0.48,
            min_confidence=0.58,
            min_agreement=0.52,
            min_odds=1.30,
            max_odds=8.50,
        )
        self.ranker = ValueBetRanker()

    def enrich_market_record(self, record: dict, bankroll: float = 1000.0) -> dict:
        probability = float(record.get("probability", 0.0))
        bookmaker_odds = float(record.get("bookmaker_odds", 0.0))
        confidence = float(record.get("confidence", 0.0))
        agreement = float(record.get("agreement", 0.0))
        opening_odds = float(record.get("opening_odds", bookmaker_odds))

        fair_odds = self.ev.fair_odds(probability)
        implied_probability = self.ev.implied_probability(bookmaker_odds)
        probability_edge = self.ev.probability_edge(probability, bookmaker_odds)
        ev_value = self.ev.expected_value(probability, bookmaker_odds)
        clv_potential = self.clv.estimate_clv_edge(
            probability, opening_odds, bookmaker_odds
        )
        movement_score = self.clv.odds_movement_score(opening_odds, bookmaker_odds)

        decision = self.filter.decide(
            probability=probability,
            odds=bookmaker_odds,
            ev=ev_value,
            confidence=confidence,
            agreement=agreement,
        )

        stake = self.stake_policy.suggest(
            bankroll=bankroll,
            probability=probability,
            odds=bookmaker_odds,
            confidence=confidence,
        )

        enriched = dict(record)
        enriched["fair_odds"] = round(fair_odds, 4)
        enriched["implied_probability"] = round(implied_probability, 6)
        enriched["probability_edge"] = round(probability_edge, 6)
        enriched["ev"] = round(ev_value, 6)
        enriched["clv_potential"] = round(clv_potential, 6)
        enriched["odds_movement_score"] = round(movement_score, 6)
        enriched["decision"] = decision
        enriched["suggested_stake"] = round(stake["stake"], 2)
        enriched["stake_cap_pct"] = round(stake["max_cap_pct"], 4)

        return enriched

    def rank(self, rows: list[dict]) -> list[dict]:
        return self.ranker.sort(rows)

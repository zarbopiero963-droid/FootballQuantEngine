from __future__ import annotations

from quant.markets.market_odds_mapper import MarketOddsMapper
from quant.markets.market_probabilities import MarketProbabilities
from quant.markets.market_value_engine import MarketValueEngine
from quant.services.quant_job_runner import QuantJobRunner


class MarketExpansionEngine:

    def __init__(self):
        self.quant_runner = QuantJobRunner()
        self.market_probabilities = MarketProbabilities()
        self.odds_mapper = MarketOddsMapper()
        self.value_engine = MarketValueEngine()

    def _derive_lambda_pair(self, record: dict) -> tuple[float, float]:
        home_xg = float(record.get("expected_goals_home", 0.0) or 0.0)
        away_xg = float(record.get("expected_goals_away", 0.0) or 0.0)

        if home_xg > 0.0 and away_xg > 0.0:
            return home_xg, away_xg

        probability = float(record.get("probability", 0.33) or 0.33)
        confidence = float(record.get("confidence", 0.50) or 0.50)

        lambda_home = max(0.8, 1.20 + (probability - 0.33) + (confidence - 0.5))
        lambda_away = max(0.8, 1.05 + ((1.0 - probability) - 0.33))

        return lambda_home, lambda_away

    def _group_by_fixture(self, records: list[dict]) -> dict:
        grouped = {}

        for row in records:
            fixture_id = str(row.get("fixture_id", ""))
            if fixture_id not in grouped:
                grouped[fixture_id] = []
            grouped[fixture_id].append(row)

        return grouped

    def _fixture_meta(self, fixture_rows: list[dict]) -> dict:
        first = fixture_rows[0]

        bookmaker_odds = {
            "home": 0.0,
            "draw": 0.0,
            "away": 0.0,
            "over_25": float(first.get("over_25_odds", 0.0) or 0.0),
            "under_25": float(first.get("under_25_odds", 0.0) or 0.0),
            "btts_yes": float(first.get("btts_yes_odds", 0.0) or 0.0),
            "btts_no": float(first.get("btts_no_odds", 0.0) or 0.0),
        }

        for row in fixture_rows:
            market = str(row.get("market", ""))
            if market in ("home", "draw", "away"):
                bookmaker_odds[market] = float(row.get("bookmaker_odds", 0.0) or 0.0)

        return {
            "fixture_id": str(first.get("fixture_id", "")),
            "home_team": first.get("home_team", ""),
            "away_team": first.get("away_team", ""),
            "agreement": max(
                float(r.get("agreement", 0.0) or 0.0) for r in fixture_rows
            ),
            "confidence": max(
                float(r.get("confidence", 0.0) or 0.0) for r in fixture_rows
            ),
            "bookmaker_odds": bookmaker_odds,
            "lambda_pair": self._derive_lambda_pair(first),
        }

    def _build_market_rows(self, fixture_meta: dict) -> list[dict]:
        lambda_home, lambda_away = fixture_meta["lambda_pair"]
        probs = self.market_probabilities.build(lambda_home, lambda_away)
        bookmaker_odds = fixture_meta["bookmaker_odds"]

        rows = []
        for market, probability in probs.items():
            odds = self.odds_mapper.odds_for_market(bookmaker_odds, market)

            rows.append(
                {
                    "fixture_id": fixture_meta["fixture_id"],
                    "home_team": fixture_meta["home_team"],
                    "away_team": fixture_meta["away_team"],
                    "market": market,
                    "probability": round(float(probability), 6),
                    "bookmaker_odds": round(float(odds), 4),
                    "opening_odds": round(float(odds), 4),
                    "confidence": round(float(fixture_meta["confidence"]), 6),
                    "agreement": round(float(fixture_meta["agreement"]), 6),
                    "expected_goals_home": round(float(lambda_home), 4),
                    "expected_goals_away": round(float(lambda_away), 4),
                    "source": "market_expansion",
                }
            )

        return rows

    def run(self, bankroll: float = 1000.0) -> list[dict]:
        base_records = self.quant_runner.run_cycle()
        grouped = self._group_by_fixture(base_records)

        expanded = []

        for _, fixture_rows in grouped.items():
            meta = self._fixture_meta(fixture_rows)
            market_rows = self._build_market_rows(meta)

            enriched = [
                self.value_engine.enrich_market_record(row, bankroll=bankroll)
                for row in market_rows
            ]
            expanded.extend(enriched)

        return self.value_engine.rank(expanded)

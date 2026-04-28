from __future__ import annotations

from strategies.edge_detector import EdgeDetector


class ValueBetStrategy:
    def find_value_bets(
        self,
        predictions: dict[str, dict[str, float]],
        odds_data: dict[str, dict[str, float]],
        edge_detector: EdgeDetector,
    ) -> list[dict[str, object]]:
        results: list[dict[str, object]] = []

        for match_id, prediction in predictions.items():
            if match_id not in odds_data:
                continue

            odds = odds_data[match_id]

            for market, prob_key, odds_key in (
                ("home", "home_win", "home"),
                ("draw", "draw", "draw"),
                ("away", "away_win", "away"),
            ):
                prob = prediction.get(prob_key)
                odd = odds.get(odds_key)
                if edge_detector.is_value_bet(prob, odd):
                    results.append(
                        {
                            "match_id": match_id,
                            "market": market,
                            "probability": prob,
                            "odds": odd,
                        }
                    )

        return results

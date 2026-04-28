from __future__ import annotations


class MarketComparison:

    def implied_probabilities(self, odds):

        home_odds = float(odds.get("home", 0) or 0)
        draw_odds = float(odds.get("draw", 0) or 0)
        away_odds = float(odds.get("away", 0) or 0)

        raw = {
            "home_win": 1 / home_odds if home_odds > 0 else 0.0,
            "draw": 1 / draw_odds if draw_odds > 0 else 0.0,
            "away_win": 1 / away_odds if away_odds > 0 else 0.0,
        }

        total = raw["home_win"] + raw["draw"] + raw["away_win"]

        if total <= 0:
            return {
                "home_win": 0.0,
                "draw": 0.0,
                "away_win": 0.0,
            }

        return {
            "home_win": raw["home_win"] / total,
            "draw": raw["draw"] / total,
            "away_win": raw["away_win"] / total,
        }

    def edge_vs_market(self, model_probs, market_probs):

        return {
            "home_win": float(model_probs.get("home_win", 0.0))
            - float(market_probs.get("home_win", 0.0)),
            "draw": float(model_probs.get("draw", 0.0))
            - float(market_probs.get("draw", 0.0)),
            "away_win": float(model_probs.get("away_win", 0.0))
            - float(market_probs.get("away_win", 0.0)),
        }

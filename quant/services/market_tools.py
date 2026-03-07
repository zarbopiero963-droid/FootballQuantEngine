class MarketTools:

    def normalize_implied_probs_1x2(self, odds):
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
                "home_win": 1 / 3,
                "draw": 1 / 3,
                "away_win": 1 / 3,
            }

        return {
            "home_win": raw["home_win"] / total,
            "draw": raw["draw"] / total,
            "away_win": raw["away_win"] / total,
        }

    def fair_odds(self, probability):
        probability = float(probability)
        if probability <= 0:
            return 999.0
        return 1.0 / probability

    def edge(self, probability, bookmaker_odds):
        probability = float(probability)
        bookmaker_odds = float(bookmaker_odds or 0.0)
        if bookmaker_odds <= 0:
            return 0.0
        return (probability * bookmaker_odds) - 1.0

    def probability_edge_vs_market(self, model_prob, market_prob):
        return float(model_prob) - float(market_prob)

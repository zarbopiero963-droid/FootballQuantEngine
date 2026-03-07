class ManualBlendModel:

    def __init__(self):
        self.weights = {
            "poisson": 0.42,
            "elo": 0.18,
            "form": 0.10,
            "xg": 0.20,
            "market": 0.10,
        }

    def elo_to_probs(self, elo_diff):
        scaled = max(min(elo_diff / 400.0, 1.0), -1.0)

        home = 0.45 + scaled * 0.22
        away = 0.27 - scaled * 0.18
        draw = 1.0 - home - away

        draw = max(0.10, draw)
        total = home + draw + away

        return {
            "home_win": home / total,
            "draw": draw / total,
            "away_win": away / total,
        }

    def form_to_probs(self, form_diff):
        scaled = max(min(form_diff, 1.0), -1.0)

        home = 0.43 + scaled * 0.20
        away = 0.29 - scaled * 0.16
        draw = 1.0 - home - away

        draw = max(0.10, draw)
        total = home + draw + away

        return {
            "home_win": home / total,
            "draw": draw / total,
            "away_win": away / total,
        }

    def xg_to_probs(self, xg_diff):
        scaled = max(min(xg_diff / 1.5, 1.0), -1.0)

        home = 0.44 + scaled * 0.24
        away = 0.28 - scaled * 0.20
        draw = 1.0 - home - away

        draw = max(0.08, draw)
        total = home + draw + away

        return {
            "home_win": home / total,
            "draw": draw / total,
            "away_win": away / total,
        }

    def market_support_probs(self, market_probs):
        return {
            "home_win": float(market_probs.get("home_win", 1 / 3)),
            "draw": float(market_probs.get("draw", 1 / 3)),
            "away_win": float(market_probs.get("away_win", 1 / 3)),
        }

    def combine(self, poisson_probs, elo_diff, form_diff, xg_diff, market_probs):
        elo_probs = self.elo_to_probs(elo_diff)
        form_probs = self.form_to_probs(form_diff)
        xg_probs = self.xg_to_probs(xg_diff)
        market_component = self.market_support_probs(market_probs)

        result = {}
        for key in ("home_win", "draw", "away_win"):
            value = 0.0
            value += self.weights["poisson"] * float(poisson_probs.get(key, 0.0))
            value += self.weights["elo"] * float(elo_probs.get(key, 0.0))
            value += self.weights["form"] * float(form_probs.get(key, 0.0))
            value += self.weights["xg"] * float(xg_probs.get(key, 0.0))
            value += self.weights["market"] * float(market_component.get(key, 0.0))
            result[key] = value

        total = sum(result.values())
        if total <= 0:
            return {
                "home_win": 1 / 3,
                "draw": 1 / 3,
                "away_win": 1 / 3,
            }

        return {
            "home_win": result["home_win"] / total,
            "draw": result["draw"] / total,
            "away_win": result["away_win"] / total,
        }

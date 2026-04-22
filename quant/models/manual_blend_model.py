class ManualBlendModel:
    """
    Weighted blend of multiple probability-space components.

    Pre-blend lambda modifiers (rest, injuries, weather, referee) are applied
    inside DixonColesEngine before this model receives DC probabilities.

    Blend weights (must sum to 1.0):
        dc         0.35  Dixon-Coles with rho correction
        elo        0.13  ELO rating difference
        form       0.08  Recent win/draw/loss form
        xg         0.09  Hybrid expected-goals signal
        h2h        0.10  Head-to-head historical dominance
        momentum   0.07  Exponentially-weighted goal momentum
        motivation 0.05  League-table stakes (title / relegation)
        market     0.13  Market-implied probabilities
    """

    def __init__(self):
        self.weights = {
            "dc": 0.35,
            "elo": 0.13,
            "form": 0.08,
            "xg": 0.09,
            "h2h": 0.10,
            "momentum": 0.07,
            "motivation": 0.05,
            "market": 0.13,
        }

    # ------------------------------------------------------------------
    # Signal → probability converters
    # ------------------------------------------------------------------

    def _diff_to_probs(
        self,
        diff: float,
        scale: float,
        home_base: float,
        away_base: float,
        home_range: float,
        away_range: float,
        draw_floor: float,
    ) -> dict:
        scaled = max(min(diff / scale, 1.0), -1.0)
        home = home_base + scaled * home_range
        away = away_base - scaled * away_range
        draw = max(draw_floor, 1.0 - home - away)
        total = home + draw + away
        return {
            "home_win": home / total,
            "draw": draw / total,
            "away_win": away / total,
        }

    def elo_to_probs(self, elo_diff: float) -> dict:
        return self._diff_to_probs(elo_diff, 400.0, 0.45, 0.27, 0.22, 0.18, 0.10)

    def form_to_probs(self, form_diff: float) -> dict:
        return self._diff_to_probs(form_diff, 1.0, 0.43, 0.29, 0.20, 0.16, 0.10)

    def xg_to_probs(self, xg_diff: float) -> dict:
        return self._diff_to_probs(xg_diff, 1.5, 0.44, 0.28, 0.24, 0.20, 0.08)

    def h2h_to_probs(self, h2h_diff: float) -> dict:
        """h2h_diff in [-1, 1]."""
        return self._diff_to_probs(h2h_diff, 1.0, 0.43, 0.28, 0.18, 0.14, 0.10)

    def momentum_to_probs(self, momentum_diff: float) -> dict:
        """momentum_diff = difference in recent scoring rate (goals/game)."""
        return self._diff_to_probs(momentum_diff, 2.0, 0.43, 0.28, 0.18, 0.14, 0.10)

    def motivation_to_probs(self, motivation_diff: float) -> dict:
        """motivation_diff in [-0.5, 0.5]."""
        return self._diff_to_probs(motivation_diff, 0.5, 0.42, 0.29, 0.12, 0.10, 0.12)

    def market_support_probs(self, market_probs: dict) -> dict:
        return {
            "home_win": float(market_probs.get("home_win", 1 / 3)),
            "draw": float(market_probs.get("draw", 1 / 3)),
            "away_win": float(market_probs.get("away_win", 1 / 3)),
        }

    # ------------------------------------------------------------------
    # Main combine
    # ------------------------------------------------------------------

    def combine(
        self,
        dc_probs: dict,
        elo_diff: float,
        form_diff: float,
        xg_diff: float,
        market_probs: dict,
        h2h_diff: float = 0.0,
        momentum_diff: float = 0.0,
        motivation_diff: float = 0.0,
    ) -> dict:
        components = {
            "dc": dc_probs,
            "elo": self.elo_to_probs(elo_diff),
            "form": self.form_to_probs(form_diff),
            "xg": self.xg_to_probs(xg_diff),
            "h2h": self.h2h_to_probs(h2h_diff),
            "momentum": self.momentum_to_probs(momentum_diff),
            "motivation": self.motivation_to_probs(motivation_diff),
            "market": self.market_support_probs(market_probs),
        }

        result: dict = {}
        for key in ("home_win", "draw", "away_win"):
            value = sum(
                self.weights[name] * float(probs.get(key, 0.0))
                for name, probs in components.items()
            )
            result[key] = value

        total = sum(result.values())
        if total <= 0:
            return {"home_win": 1 / 3, "draw": 1 / 3, "away_win": 1 / 3}

        return {
            "home_win": result["home_win"] / total,
            "draw": result["draw"] / total,
            "away_win": result["away_win"] / total,
        }

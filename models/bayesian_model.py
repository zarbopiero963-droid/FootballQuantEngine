class BayesianModel:

    def predict(self, home_attack, home_defense, away_attack, away_defense):

        home_strength = home_attack / max(away_defense, 0.0001)
        away_strength = away_attack / max(home_defense, 0.0001)
        draw_strength = 1.0

        total = home_strength + draw_strength + away_strength

        if total <= 0:
            return {
                "home_win": 0.0,
                "draw": 0.0,
                "away_win": 0.0,
            }

        return {
            "home_win": home_strength / total,
            "draw": draw_strength / total,
            "away_win": away_strength / total,
        }

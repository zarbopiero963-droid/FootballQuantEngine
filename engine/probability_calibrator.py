class ProbabilityCalibrator:

    def __init__(self, shrink=0.12):

        self.shrink = shrink

    def calibrate(self, probs):

        home = float(probs.get("home_win", 0.0))
        draw = float(probs.get("draw", 0.0))
        away = float(probs.get("away_win", 0.0))

        total = home + draw + away
        if total <= 0:
            return {
                "home_win": 1 / 3,
                "draw": 1 / 3,
                "away_win": 1 / 3,
            }

        home /= total
        draw /= total
        away /= total

        uniform = 1 / 3

        home = (1 - self.shrink) * home + self.shrink * uniform
        draw = (1 - self.shrink) * draw + self.shrink * uniform
        away = (1 - self.shrink) * away + self.shrink * uniform

        total = home + draw + away

        return {
            "home_win": home / total,
            "draw": draw / total,
            "away_win": away / total,
        }

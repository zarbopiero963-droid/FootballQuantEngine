from __future__ import annotations

from models.elo_model import EloModel


class ModelPlugin:

    name = "elo"

    def __init__(self):

        self.model = EloModel()

    def predict(self, row):

        probability = self.model.probability(row["elo_diff"])

        draw_probability = 0.20
        away_probability = 1 - probability - draw_probability

        if away_probability < 0:
            away_probability = 0.0

        total = probability + draw_probability + away_probability

        if total <= 0:
            return {
                "home_win": 0.0,
                "draw": 0.0,
                "away_win": 0.0,
            }

        return {
            "home_win": probability / total,
            "draw": draw_probability / total,
            "away_win": away_probability / total,
        }

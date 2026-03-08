from __future__ import annotations

import math


class OverUnderModel:

    def __init__(self, max_goals: int = 10):
        self.max_goals = int(max_goals)

    def _poisson_pmf(self, goals: int, lam: float) -> float:
        lam = max(0.01, float(lam))
        return (math.exp(-lam) * (lam**goals)) / math.factorial(goals)

    def _score_matrix(
        self, lambda_home: float, lambda_away: float
    ) -> list[list[float]]:
        matrix = []

        home_probs = [
            self._poisson_pmf(i, lambda_home) for i in range(self.max_goals + 1)
        ]
        away_probs = [
            self._poisson_pmf(i, lambda_away) for i in range(self.max_goals + 1)
        ]

        home_total = sum(home_probs) or 1.0
        away_total = sum(away_probs) or 1.0

        home_probs = [x / home_total for x in home_probs]
        away_probs = [x / away_total for x in away_probs]

        for hg in range(self.max_goals + 1):
            row = []
            for ag in range(self.max_goals + 1):
                row.append(home_probs[hg] * away_probs[ag])
            matrix.append(row)

        return matrix

    def probabilities(
        self, lambda_home: float, lambda_away: float, line: float = 2.5
    ) -> dict:
        matrix = self._score_matrix(lambda_home, lambda_away)

        over_prob = 0.0
        under_prob = 0.0

        for hg, row in enumerate(matrix):
            for ag, prob in enumerate(row):
                total_goals = hg + ag

                if total_goals > line:
                    over_prob += prob
                else:
                    under_prob += prob

        total = over_prob + under_prob
        if total <= 0:
            return {
                "over": 0.5,
                "under": 0.5,
            }

        return {
            "over": over_prob / total,
            "under": under_prob / total,
        }

    def fair_odds(self, probability: float) -> float:
        probability = float(probability)
        if probability <= 0.0:
            return 999.0
        return 1.0 / probability

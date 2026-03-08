from __future__ import annotations

import math


class BTTSModel:

    def __init__(self, max_goals: int = 10):
        self.max_goals = int(max_goals)

    def _poisson_pmf(self, goals: int, lam: float) -> float:
        lam = max(0.01, float(lam))
        return (math.exp(-lam) * (lam**goals)) / math.factorial(goals)

    def probabilities(self, lambda_home: float, lambda_away: float) -> dict:
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

        yes_prob = 0.0
        no_prob = 0.0

        for hg in range(self.max_goals + 1):
            for ag in range(self.max_goals + 1):
                prob = home_probs[hg] * away_probs[ag]

                if hg > 0 and ag > 0:
                    yes_prob += prob
                else:
                    no_prob += prob

        total = yes_prob + no_prob
        if total <= 0:
            return {
                "yes": 0.5,
                "no": 0.5,
            }

        return {
            "yes": yes_prob / total,
            "no": no_prob / total,
        }

    def fair_odds(self, probability: float) -> float:
        probability = float(probability)
        if probability <= 0.0:
            return 999.0
        return 1.0 / probability

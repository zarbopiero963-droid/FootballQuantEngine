"""
Correct Score Model
===================
Derives exact-score probabilities directly from the Poisson score matrix
already computed by PoissonEngine.

Returns the top-N most likely scores ranked by probability, plus
fair odds for each.
"""

from __future__ import annotations

import math


class CorrectScoreModel:

    def __init__(self, max_goals: int = 8, top_n: int = 12):
        self.max_goals = max_goals
        self.top_n = top_n

    def probabilities(
        self,
        score_matrix: list[list[float]],
        top_n: int | None = None,
    ) -> list[dict]:
        """
        Args:
            score_matrix: (max_goals+1) × (max_goals+1) probability matrix
                          from PoissonEngine.score_matrix()
            top_n: how many scores to return (defaults to self.top_n)

        Returns a list of dicts sorted by probability descending:
            [
                {'score': '1-0', 'home_goals': 1, 'away_goals': 0,
                 'probability': 0.142, 'fair_odds': 7.04},
                ...
            ]
        """
        top_n = top_n or self.top_n
        scores = []

        for hg, row in enumerate(score_matrix):
            for ag, prob in enumerate(row):
                if prob <= 0:
                    continue
                scores.append(
                    {
                        "score": f"{hg}-{ag}",
                        "home_goals": hg,
                        "away_goals": ag,
                        "probability": round(prob, 5),
                        "fair_odds": round(1.0 / prob, 2) if prob > 0 else 999.0,
                    }
                )

        scores.sort(key=lambda x: x["probability"], reverse=True)  # type: ignore[arg-type,return-value]
        return scores[:top_n]

    def top_scores_dict(
        self,
        score_matrix: list[list[float]],
        top_n: int | None = None,
    ) -> dict[str, float]:
        """
        Returns {score_string: probability} for the top-N scores.
        Useful for serialisation / DB storage.
        """
        return {
            s["score"]: s["probability"]
            for s in self.probabilities(score_matrix, top_n)
        }

    def halftime_probabilities(
        self,
        lambda_ht_home: float,
        lambda_ht_away: float,
        top_n: int | None = None,
    ) -> list[dict]:
        """
        Build a halftime correct-score distribution from HT lambdas.
        Uses same Poisson approach with max_goals capped at 4 for HT.
        """
        max_ht = 4
        matrix = self._build_matrix(lambda_ht_home, lambda_ht_away, max_ht)
        return self.probabilities(matrix, top_n or self.top_n)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _poisson_pmf(self, k: int, lam: float) -> float:
        lam = max(0.01, lam)
        return math.exp(-lam) * (lam**k) / math.factorial(min(k, 170))

    def _build_matrix(
        self, lam_h: float, lam_a: float, max_g: int
    ) -> list[list[float]]:
        raw = []
        for hg in range(max_g + 1):
            row = []
            for ag in range(max_g + 1):
                row.append(self._poisson_pmf(hg, lam_h) * self._poisson_pmf(ag, lam_a))
            raw.append(row)

        # Normalise so probabilities sum to 1
        total = sum(p for row in raw for p in row) or 1.0
        return [[p / total for p in row] for row in raw]

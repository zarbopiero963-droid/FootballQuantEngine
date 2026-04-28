from __future__ import annotations

from math import exp, factorial


def poisson_pmf(k, lam):

    return exp(-lam) * (lam**k) / factorial(k)


class ProbabilityMarkets:

    def score_matrix(self, lambda_home, lambda_away, max_goals=10):

        matrix = []

        for i in range(max_goals + 1):
            row = []
            for j in range(max_goals + 1):
                row.append(poisson_pmf(i, lambda_home) * poisson_pmf(j, lambda_away))
            matrix.append(row)

        return matrix

    def over_under_25(self, lambda_home, lambda_away):

        matrix = self.score_matrix(lambda_home, lambda_away)

        under = 0.0
        over = 0.0

        for i, row in enumerate(matrix):
            for j, value in enumerate(row):
                if i + j <= 2:
                    under += value
                else:
                    over += value

        return {
            "under_2_5": under,
            "over_2_5": over,
        }

    def btts(self, lambda_home, lambda_away):

        p_home_zero = poisson_pmf(0, lambda_home)
        p_away_zero = poisson_pmf(0, lambda_away)
        p_both_zero = p_home_zero * p_away_zero

        yes = 1 - p_home_zero - p_away_zero + p_both_zero
        no = 1 - yes

        return {
            "btts_yes": yes,
            "btts_no": no,
        }

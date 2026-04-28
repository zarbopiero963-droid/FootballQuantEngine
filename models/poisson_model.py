from __future__ import annotations

from scipy.stats import poisson


class PoissonModel:

    def predict(self, home_attack, home_defense, away_attack, away_defense):

        lambda_home = home_attack * away_defense
        lambda_away = away_attack * home_defense

        max_goals = 6

        home_probs = [poisson.pmf(i, lambda_home) for i in range(max_goals)]
        away_probs = [poisson.pmf(i, lambda_away) for i in range(max_goals)]

        home_win = 0
        draw = 0
        away_win = 0

        for i in range(max_goals):
            for j in range(max_goals):

                p = home_probs[i] * away_probs[j]

                if i > j:
                    home_win += p
                elif i == j:
                    draw += p
                else:
                    away_win += p

        return {"home_win": home_win, "draw": draw, "away_win": away_win}

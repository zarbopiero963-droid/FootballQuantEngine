from scipy.stats import poisson


class BivariatePoisson:

    def predict(self, lambda_home, lambda_away, rho=0.05):

        max_goals = 6

        home_win = 0
        draw = 0
        away_win = 0

        for i in range(max_goals):
            for j in range(max_goals):

                p_home = poisson.pmf(i, lambda_home)
                p_away = poisson.pmf(j, lambda_away)

                correlation = 1 + rho if i == j else 1

                p = p_home * p_away * correlation

                if i > j:
                    home_win += p
                elif i == j:
                    draw += p
                else:
                    away_win += p

        total = home_win + draw + away_win

        return {
            "home_win": home_win / total,
            "draw": draw / total,
            "away_win": away_win / total,
        }

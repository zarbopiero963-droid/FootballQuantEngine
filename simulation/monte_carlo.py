import numpy as np


class MonteCarloSimulator:

    def simulate(self, lambda_home, lambda_away, simulations=20000):

        home_goals = np.random.poisson(lambda_home, simulations)
        away_goals = np.random.poisson(lambda_away, simulations)

        home_win = (home_goals > away_goals).mean()
        draw = (home_goals == away_goals).mean()
        away_win = (home_goals < away_goals).mean()

        return {"home_win": home_win, "draw": draw, "away_win": away_win}

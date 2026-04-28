from __future__ import annotations

from models.bivariate_poisson import BivariatePoisson


class ModelPlugin:

    name = "bivariate_poisson"

    def __init__(self):

        self.model = BivariatePoisson()

    def predict(self, row):

        lambda_home = row["home_attack"] * row["away_defense"]
        lambda_away = row["away_attack"] * row["home_defense"]

        return self.model.predict(
            lambda_home=lambda_home,
            lambda_away=lambda_away,
            rho=0.05,
        )

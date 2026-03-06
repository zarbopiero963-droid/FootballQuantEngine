from models.poisson_model import PoissonModel


class ModelPlugin:

    name = "poisson"

    def __init__(self):

        self.model = PoissonModel()

    def predict(self, row):

        return self.model.predict(
            home_attack=row["home_attack"],
            home_defense=row["home_defense"],
            away_attack=row["away_attack"],
            away_defense=row["away_defense"],
        )

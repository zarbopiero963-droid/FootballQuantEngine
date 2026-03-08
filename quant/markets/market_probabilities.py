from __future__ import annotations

from quant.markets.btts_model import BTTSModel
from quant.markets.over_under_model import OverUnderModel


class MarketProbabilities:

    def __init__(self):
        self.ou = OverUnderModel()
        self.btts = BTTSModel()

    def build(self, lambda_home: float, lambda_away: float) -> dict:
        ou_25 = self.ou.probabilities(lambda_home, lambda_away, line=2.5)
        btts = self.btts.probabilities(lambda_home, lambda_away)

        return {
            "over_25": ou_25["over"],
            "under_25": ou_25["under"],
            "btts_yes": btts["yes"],
            "btts_no": btts["no"],
        }

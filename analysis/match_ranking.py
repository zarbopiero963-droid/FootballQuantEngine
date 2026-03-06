import pandas as pd


class MatchRanking:

    def rank(self, value_bets):

        if not value_bets:
            return pd.DataFrame()

        df = pd.DataFrame(value_bets)

        if "probability" not in df.columns or "odds" not in df.columns:
            return df

        df["edge"] = df["probability"] - (1 / df["odds"])

        return df.sort_values(by="edge", ascending=False)

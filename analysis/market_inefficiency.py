import pandas as pd


class MarketInefficiencyScanner:

    def scan(self, predictions_df):

        if predictions_df.empty:
            return pd.DataFrame()

        df = predictions_df.copy()

        required = {
            "home_win",
            "draw",
            "away_win",
        }

        if not required.issubset(df.columns):
            return pd.DataFrame()

        rows = []

        for _, row in df.iterrows():

            probs = {
                "home": row["home_win"],
                "draw": row["draw"],
                "away": row["away_win"],
            }

            best_market = max(probs, key=probs.get)
            best_probability = probs[best_market]

            inefficiency_score = best_probability - (1 / 3)

            rows.append(
                {
                    "team": row.get("team"),
                    "best_market": best_market,
                    "best_probability": best_probability,
                    "inefficiency_score": inefficiency_score,
                }
            )

        result = pd.DataFrame(rows)

        return result.sort_values(
            by="inefficiency_score",
            ascending=False,
        )

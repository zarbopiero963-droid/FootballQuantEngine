import pandas as pd

from database.db_manager import connect


class BacktestEngine:

    def load_predictions(self):

        conn = connect()

        df = pd.read_sql_query(
            """
            SELECT
                fixture_id,
                model,
                home_prob,
                draw_prob,
                away_prob,
                edge,
                created_at
            FROM predictions
            """,
            conn,
        )

        conn.close()

        return df

    def load_results(self):

        conn = connect()

        df = pd.read_sql_query(
            """
            SELECT
                fixture_id,
                home_goals,
                away_goals
            FROM fixtures
            WHERE status = 'FT'
            """,
            conn,
        )

        conn.close()

        return df

    def run(self):

        predictions = self.load_predictions()
        results = self.load_results()

        if predictions.empty or results.empty:
            return pd.DataFrame()

        merged = predictions.merge(results, on="fixture_id", how="inner")

        merged["actual_home_win"] = (
            merged["home_goals"] > merged["away_goals"]
        ).astype(int)

        merged["actual_draw"] = (merged["home_goals"] == merged["away_goals"]).astype(
            int
        )

        merged["actual_away_win"] = (
            merged["home_goals"] < merged["away_goals"]
        ).astype(int)

        return merged

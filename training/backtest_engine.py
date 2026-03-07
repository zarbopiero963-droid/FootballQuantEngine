import pandas as pd

from database.db_manager import connect


class BacktestEngine:

    def load_predictions_with_results(self, date_from=None, date_to=None, league=None):

        conn = connect()

        query = """
        SELECT
            p.fixture_id,
            p.home_prob,
            p.draw_prob,
            p.away_prob,
            f.league,
            f.match_date,
            f.home_goals,
            f.away_goals,
            oh.home_odds,
            oh.draw_odds,
            oh.away_odds
        FROM predictions p
        JOIN fixtures f
            ON p.fixture_id = f.fixture_id
        LEFT JOIN (
            SELECT o1.*
            FROM odds_history o1
            JOIN (
                SELECT fixture_id, MAX(timestamp) AS max_ts
                FROM odds_history
                GROUP BY fixture_id
            ) o2
                ON o1.fixture_id = o2.fixture_id
               AND o1.timestamp = o2.max_ts
        ) oh
            ON p.fixture_id = oh.fixture_id
        WHERE f.status = 'FT'
        """

        params = []

        if date_from:
            query += " AND f.match_date >= ?"
            params.append(date_from)

        if date_to:
            query += " AND f.match_date <= ?"
            params.append(date_to)

        if league:
            query += " AND f.league = ?"
            params.append(league)

        df = pd.read_sql_query(query, conn, params=params)
        conn.close()

        if df.empty:
            return df

        df["actual_home_win"] = (df["home_goals"] > df["away_goals"]).astype(int)
        df["actual_draw"] = (df["home_goals"] == df["away_goals"]).astype(int)
        df["actual_away_win"] = (df["home_goals"] < df["away_goals"]).astype(int)

        df["home_odds"] = df["home_odds"].fillna(2.0)
        df["draw_odds"] = df["draw_odds"].fillna(3.0)
        df["away_odds"] = df["away_odds"].fillna(2.5)

        return df

    def run(self, date_from=None, date_to=None, league=None):

        return self.load_predictions_with_results(
            date_from=date_from,
            date_to=date_to,
            league=league,
        )

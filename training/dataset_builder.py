import pandas as pd

from analysis.betfair_leagues import is_betfair_league
from database.db_manager import connect


class DatasetBuilder:

    def load_fixtures(self):

        conn = connect()

        fixtures_df = pd.read_sql_query(
            """
            SELECT
                fixture_id,
                league,
                season,
                home,
                away,
                match_date,
                home_goals,
                away_goals,
                status
            FROM fixtures
            WHERE status = 'FT'
            """,
            conn,
        )

        conn.close()

        return fixtures_df

    def build_training_dataset(self):

        fixtures_df = self.load_fixtures()

        if fixtures_df.empty:
            return pd.DataFrame()

        fixtures_df = fixtures_df[fixtures_df["league"].apply(is_betfair_league)].copy()

        if fixtures_df.empty:
            return pd.DataFrame()

        fixtures_df["target_home_win"] = (
            fixtures_df["home_goals"] > fixtures_df["away_goals"]
        ).astype(int)

        fixtures_df["target_draw"] = (
            fixtures_df["home_goals"] == fixtures_df["away_goals"]
        ).astype(int)

        fixtures_df["target_away_win"] = (
            fixtures_df["home_goals"] < fixtures_df["away_goals"]
        ).astype(int)

        return fixtures_df

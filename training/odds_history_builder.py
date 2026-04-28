from __future__ import annotations

import pandas as pd

from database.db_manager import connect


class OddsHistoryBuilder:

    def load_odds_history(self):

        conn = connect()

        df = pd.read_sql_query(
            """
            SELECT
                fixture_id,
                timestamp,
                home_odds,
                draw_odds,
                away_odds
            FROM odds_history
            ORDER BY timestamp ASC
            """,
            conn,
        )

        conn.close()

        return df

from __future__ import annotations

import pandas as pd

from database.db_manager import connect


class ClosingLineBuilder:

    def load_closing_lines(self):

        conn = connect()

        df = pd.read_sql_query(
            """
            SELECT
                fixture_id,
                home_odds,
                draw_odds,
                away_odds,
                timestamp
            FROM odds_history
            ORDER BY timestamp ASC
            """,
            conn,
        )

        conn.close()

        if df.empty:
            return pd.DataFrame()

        closing = (
            df.sort_values("timestamp")
            .groupby("fixture_id")
            .agg(
                closing_home_odds=("home_odds", "last"),
                closing_draw_odds=("draw_odds", "last"),
                closing_away_odds=("away_odds", "last"),
            )
            .reset_index()
        )

        return closing

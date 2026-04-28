from __future__ import annotations

from datetime import datetime

from database.db_manager import connect


class OddsRepository:

    def save_odds(
        self,
        fixture_id: int | str,
        home_odds: float,
        draw_odds: float,
        away_odds: float,
    ) -> None:

        conn = connect()

        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO odds_history(
                fixture_id,
                timestamp,
                home_odds,
                draw_odds,
                away_odds
            )
            VALUES(?,?,?,?,?)
            """,
            (
                fixture_id,
                datetime.utcnow().isoformat(),
                home_odds,
                draw_odds,
                away_odds,
            ),
        )

        conn.commit()
        conn.close()

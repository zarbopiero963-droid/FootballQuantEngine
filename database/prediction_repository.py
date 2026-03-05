from datetime import datetime

from database.db_manager import connect


class PredictionRepository:

    def save_prediction(self, fixture_id, model, home_prob, draw_prob, away_prob, edge):

        conn = connect()

        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO predictions(
                fixture_id,
                model,
                home_prob,
                draw_prob,
                away_prob,
                edge,
                created_at
            )
            VALUES(?,?,?,?,?,?,?)
            """,
            (
                fixture_id,
                model,
                home_prob,
                draw_prob,
                away_prob,
                edge,
                datetime.utcnow().isoformat(),
            ),
        )

        conn.commit()
        conn.close()

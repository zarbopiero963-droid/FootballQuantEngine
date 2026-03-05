from database.db_manager import connect


class FixturesRepository:

    def save_fixture(self, fixture):

        conn = connect()

        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT OR REPLACE INTO fixtures(
                fixture_id,
                league,
                season,
                home,
                away,
                match_date,
                home_goals,
                away_goals,
                status
            )
            VALUES(?,?,?,?,?,?,?,?,?)
            """,
            (
                fixture.get("fixture_id"),
                fixture.get("league"),
                fixture.get("season"),
                fixture.get("home"),
                fixture.get("away"),
                fixture.get("match_date"),
                fixture.get("home_goals"),
                fixture.get("away_goals"),
                fixture.get("status"),
            ),
        )

        conn.commit()
        conn.close()

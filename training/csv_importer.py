import pandas as pd

from database.fixtures_repository import FixturesRepository
from database.odds_repository import OddsRepository


class CsvImporter:

    def __init__(self):

        self.fixtures_repo = FixturesRepository()
        self.odds_repo = OddsRepository()

    def import_matches(self, filepath):

        df = pd.read_csv(filepath)

        required = [
            "fixture_id",
            "league",
            "season",
            "home",
            "away",
            "match_date",
            "home_goals",
            "away_goals",
            "status",
        ]

        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing column: {col}")

        for _, row in df.iterrows():
            self.fixtures_repo.save_fixture(
                {
                    "fixture_id": int(row["fixture_id"]),
                    "league": row["league"],
                    "season": int(row["season"]),
                    "home": row["home"],
                    "away": row["away"],
                    "match_date": row["match_date"],
                    "home_goals": int(row["home_goals"]),
                    "away_goals": int(row["away_goals"]),
                    "status": row["status"],
                }
            )

    def import_odds(self, filepath):

        df = pd.read_csv(filepath)

        required = [
            "fixture_id",
            "home_odds",
            "draw_odds",
            "away_odds",
        ]

        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing column: {col}")

        for _, row in df.iterrows():
            self.odds_repo.save_odds(
                fixture_id=int(row["fixture_id"]),
                home_odds=float(row["home_odds"]),
                draw_odds=float(row["draw_odds"]),
                away_odds=float(row["away_odds"]),
            )

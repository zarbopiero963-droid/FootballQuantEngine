from database.fixtures_repository import FixturesRepository
from database.odds_repository import OddsRepository
from training.import_validator import ImportValidator


class CsvImporter:

    def __init__(self):

        self.fixtures_repo = FixturesRepository()
        self.odds_repo = OddsRepository()
        self.validator = ImportValidator()

    def import_matches(self, filepath):

        validation = self.validator.load_and_validate_matches_csv(filepath)

        if not validation["ok"]:
            raise ValueError(
                f"Missing columns in matches CSV: {validation['missing_columns']}"
            )

        df = validation["dataframe"]

        imported = 0

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
            imported += 1

        return {
            "type": "matches",
            "imported_rows": imported,
            "row_count": validation["row_count"],
        }

    def import_odds(self, filepath):

        validation = self.validator.load_and_validate_odds_csv(filepath)

        if not validation["ok"]:
            raise ValueError(
                f"Missing columns in odds CSV: {validation['missing_columns']}"
            )

        df = validation["dataframe"]

        imported = 0

        for _, row in df.iterrows():
            self.odds_repo.save_odds(
                fixture_id=int(row["fixture_id"]),
                home_odds=float(row["home_odds"]),
                draw_odds=float(row["draw_odds"]),
                away_odds=float(row["away_odds"]),
            )
            imported += 1

        return {
            "type": "odds",
            "imported_rows": imported,
            "row_count": validation["row_count"],
        }

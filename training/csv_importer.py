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

        for row_idx, row in df.iterrows():
            try:
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
            except (ValueError, TypeError) as exc:
                raise ValueError(
                    f"Invalid data at row {row_idx + 1}: {exc}"
                ) from exc
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

        for row_idx, row in df.iterrows():
            try:
                home_odds = float(row["home_odds"])
                draw_odds = float(row["draw_odds"])
                away_odds = float(row["away_odds"])
                if home_odds <= 1.0 or draw_odds <= 1.0 or away_odds <= 1.0:
                    raise ValueError(
                        f"Odds must be > 1.0 (got home={home_odds}, draw={draw_odds}, away={away_odds})"
                    )
                self.odds_repo.save_odds(
                    fixture_id=int(row["fixture_id"]),
                    home_odds=home_odds,
                    draw_odds=draw_odds,
                    away_odds=away_odds,
                )
            except (ValueError, TypeError) as exc:
                raise ValueError(
                    f"Invalid odds data at row {row_idx + 1}: {exc}"
                ) from exc
            imported += 1

        return {
            "type": "odds",
            "imported_rows": imported,
            "row_count": validation["row_count"],
        }

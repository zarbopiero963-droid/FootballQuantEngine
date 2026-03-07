import pandas as pd


class ImportValidator:

    MATCHES_REQUIRED_COLUMNS = [
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

    ODDS_REQUIRED_COLUMNS = [
        "fixture_id",
        "home_odds",
        "draw_odds",
        "away_odds",
    ]

    def validate_matches_df(self, df):

        missing = [
            col for col in self.MATCHES_REQUIRED_COLUMNS if col not in df.columns
        ]

        return {
            "ok": len(missing) == 0,
            "missing_columns": missing,
            "row_count": len(df.index),
        }

    def validate_odds_df(self, df):

        missing = [col for col in self.ODDS_REQUIRED_COLUMNS if col not in df.columns]

        return {
            "ok": len(missing) == 0,
            "missing_columns": missing,
            "row_count": len(df.index),
        }

    def load_and_validate_matches_csv(self, filepath):

        df = pd.read_csv(filepath)
        result = self.validate_matches_df(df)
        result["dataframe"] = df
        return result

    def load_and_validate_odds_csv(self, filepath):

        df = pd.read_csv(filepath)
        result = self.validate_odds_df(df)
        result["dataframe"] = df
        return result

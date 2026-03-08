from __future__ import annotations

import pandas as pd


class TimeSplitter:

    def ensure_datetime(
        self, df: pd.DataFrame, date_col: str = "match_date"
    ) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()

        out = df.copy()
        out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
        out = out.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)
        return out

    def split_train_validation_test(
        self,
        df: pd.DataFrame,
        date_col: str = "match_date",
        train_ratio: float = 0.6,
        validation_ratio: float = 0.2,
    ) -> dict:
        out = self.ensure_datetime(df, date_col=date_col)

        if out.empty:
            empty = pd.DataFrame()
            return {
                "train": empty,
                "validation": empty,
                "test": empty,
            }

        n = len(out.index)
        train_end = max(1, int(n * train_ratio))
        validation_end = max(train_end + 1, int(n * (train_ratio + validation_ratio)))

        validation_end = min(validation_end, n)

        train = out.iloc[:train_end].reset_index(drop=True)
        validation = out.iloc[train_end:validation_end].reset_index(drop=True)
        test = out.iloc[validation_end:].reset_index(drop=True)

        return {
            "train": train,
            "validation": validation,
            "test": test,
        }

    def rolling_windows(
        self,
        df: pd.DataFrame,
        date_col: str = "match_date",
        train_size: int = 50,
        test_size: int = 20,
        step: int = 20,
    ) -> list[dict]:
        out = self.ensure_datetime(df, date_col=date_col)

        if out.empty:
            return []

        train_size = max(1, int(train_size))
        test_size = max(1, int(test_size))
        step = max(1, int(step))

        windows = []
        start = 0
        total = len(out.index)

        while start + train_size + test_size <= total:
            train = out.iloc[start : start + train_size].reset_index(drop=True)
            test = out.iloc[
                start + train_size : start + train_size + test_size
            ].reset_index(drop=True)

            windows.append(
                {
                    "train": train,
                    "test": test,
                    "train_start": str(train[date_col].min()),
                    "train_end": str(train[date_col].max()),
                    "test_start": str(test[date_col].min()),
                    "test_end": str(test[date_col].max()),
                }
            )

            start += step

        return windows

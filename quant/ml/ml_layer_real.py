from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from sklearn.linear_model import LogisticRegression
except Exception:
    LogisticRegression = None


class MLLayerReal:

    FEATURE_COLUMNS = [
        "elo_diff",
        "form_diff",
        "xg_diff",
        "xga_diff",
        "xpts_diff",
        "shots_diff",
        "shots_on_target_diff",
        "form_score_diff",
        "bookmaker_odds_home",
        "bookmaker_odds_draw",
        "bookmaker_odds_away",
    ]

    def __init__(self):
        self.model = None
        self.is_fitted = False

    def _build_dataframe(self, rows: list[dict]) -> pd.DataFrame:
        if not rows:
            return pd.DataFrame(columns=self.FEATURE_COLUMNS + ["target"])

        df = pd.DataFrame(rows).copy()

        for col in self.FEATURE_COLUMNS:
            if col not in df.columns:
                df[col] = 0.0

        if "result" not in df.columns:
            df["result"] = "D"

        df["target"] = (df["result"].astype(str) == "H").astype(int)

        return df[self.FEATURE_COLUMNS + ["target"]].copy()

    def fit(self, rows: list[dict]) -> None:
        df = self._build_dataframe(rows)

        if df.empty or LogisticRegression is None:
            self.model = None
            self.is_fitted = False
            return

        X = df[self.FEATURE_COLUMNS].astype(float).values
        y = df["target"].astype(int).values

        if len(set(y.tolist())) < 2:
            self.model = None
            self.is_fitted = False
            return

        self.model = LogisticRegression(max_iter=1000)
        self.model.fit(X, y)
        self.is_fitted = True

    def predict_home_probability(self, row: dict) -> float:
        if not self.is_fitted or self.model is None:
            return 0.50

        values = []
        for col in self.FEATURE_COLUMNS:
            values.append(float(row.get(col, 0.0) or 0.0))

        X = np.asarray([values], dtype=float)
        prob = float(self.model.predict_proba(X)[0, 1])

        return round(float(min(max(prob, 0.0001), 0.9999)), 6)

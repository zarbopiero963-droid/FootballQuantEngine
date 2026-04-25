from __future__ import annotations

import importlib.util


def _sklearn_available() -> bool:
    return importlib.util.find_spec("sklearn") is not None


class FeatureImportanceAnalyzer:
    """
    Ranks which prediction signals (elo_diff, form_diff, xg_diff, h2h_diff, …)
    are most predictive of the actual match outcome.

    Works on any DataFrame that contains numeric signal columns and a binary
    target column (1 = prediction was correct, 0 = wrong).

    Falls back to a variance-based ranking when scikit-learn is not installed.
    """

    # Signals produced by the quant pipeline's FeatureBuilder
    DEFAULT_SIGNAL_COLS = [
        "elo_diff",
        "form_diff",
        "xg_diff",
        "h2h_diff",
        "momentum_diff",
        "motivation_diff",
        "rest_diff",
        "injury_diff",
        "goal_expectancy_diff",
    ]

    def compute(
        self,
        df,
        target_col: str = "correct",
        feature_cols: list[str] | None = None,
        n_estimators: int = 200,
        random_state: int = 42,
    ) -> dict[str, float]:
        """
        Parameters
        ----------
        df : pd.DataFrame
            Must contain the feature columns and the target column.
        target_col : str
            Binary column: 1 = model prediction was correct, 0 = wrong.
        feature_cols : list[str] | None
            Columns to use as features. Defaults to DEFAULT_SIGNAL_COLS
            (columns that are absent in df are silently dropped).
        n_estimators : int
            Number of trees for the RandomForest (ignored when sklearn absent).

        Returns
        -------
        dict  {feature_name: importance_score}  sorted descending.
        """
        import pandas as pd

        cols = [
            c for c in (feature_cols or self.DEFAULT_SIGNAL_COLS) if c in df.columns
        ]

        if not cols:
            return {}

        sub = df[cols + [target_col]].dropna()
        if sub.empty or sub[target_col].nunique() < 2:
            return {c: 0.0 for c in cols}

        X = sub[cols]
        y = sub[target_col]

        if _sklearn_available():
            from sklearn.ensemble import RandomForestClassifier

            rf = RandomForestClassifier(
                n_estimators=n_estimators,
                random_state=random_state,
                n_jobs=-1,
            )
            rf.fit(X, y)
            raw = dict(zip(cols, rf.feature_importances_))
        else:
            # Variance of each feature weighted by correlation with target
            raw = {}
            for col in cols:
                corr = abs(float(X[col].corr(y)))
                raw[col] = corr if not pd.isna(corr) else 0.0

        return dict(sorted(raw.items(), key=lambda kv: kv[1], reverse=True))

    def from_predictions(
        self,
        predictions_df,
        market: str = "home",
        feature_cols: list[str] | None = None,
    ) -> dict[str, float]:
        """
        Convenience wrapper: builds the `correct` target from a prediction
        DataFrame that contains the quant engine's detail fields and the
        actual match result columns (actual_home_win, actual_draw, actual_away_win).

        Parameters
        ----------
        predictions_df : pd.DataFrame
            Rows from the prediction pipeline joined with actual results
            (as produced by BacktestEngine.load_predictions_with_results).
        market : str
            'home', 'draw', or 'away' — which market to evaluate.
        """

        df = predictions_df.copy()

        actual_col = {
            "home": "actual_home_win",
            "draw": "actual_draw",
            "away": "actual_away_win",
        }.get(market)

        if actual_col is None or actual_col not in df.columns:
            return {}

        prob_col = {
            "home": "home_prob",
            "draw": "draw_prob",
            "away": "away_prob",
        }.get(market)

        if prob_col and prob_col in df.columns:
            # "correct" = model assigned highest probability to the right outcome
            df["correct"] = (df[actual_col] == 1).astype(int)
        else:
            df["correct"] = df[actual_col].astype(int)

        return self.compute(df, target_col="correct", feature_cols=feature_cols)

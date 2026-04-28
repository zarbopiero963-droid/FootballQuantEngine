"""
Real AutoML model search with cross-validation and hyperparameter tuning.

Architecture
------------
1. Feature extraction  — uses only numeric columns from the dataset
2. Preprocessing       — z-score normalisation fitted on training folds only
3. Model candidates    — tries every library that is installed:
     sklearn:   RandomForest, GradientBoosting, LogisticRegression
     xgboost:   XGBClassifier
     lightgbm:  LGBMClassifier
     catboost:  CatBoostClassifier
     optuna:    tunes the best sklearn model with TPE sampler
   Pure-Python fallback (no sklearn): GaussianNB + LogisticRegression-SGD
4. Cross-validation    — StratifiedKFold (k=5) on time-sorted data
5. Selection criterion — lowest mean log-loss across folds
6. Persistence         — best model saved with joblib / pickle to MODEL_PATH
7. Return value        — rich dict with metrics, feature importances, model path

Sub-modules
-----------
training.automl_learners  — _GaussianNB, _SGDLogisticReg, _sklearn_candidates, _optuna_tune
training.automl_metrics   — metrics, preprocessing, CV runners, feature importance
"""

from __future__ import annotations

import hashlib
import logging
import os
import pickle
from typing import Any

from training.automl_learners import (
    _GaussianNB,
    _has,
    _optuna_tune,
    _SGDLogisticReg,
    _sklearn_candidates,
)
from training.automl_metrics import (
    _cv_pure_python,
    _cv_sklearn,
    _feature_importances_sklearn,
    _normalise,
)

logger = logging.getLogger(__name__)

MODEL_PATH = "outputs/automl_best_model.pkl"
_SHA256_PATH = MODEL_PATH + ".sha256"
_MIN_ROWS = 20


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def _file_sha256(path: str) -> str:
    """Return the hex SHA-256 digest of a file's contents."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _save_model(model: Any, meta: dict) -> str:
    os.makedirs(os.path.dirname(MODEL_PATH) or ".", exist_ok=True)
    payload = {"model": model, "meta": meta}
    try:
        import joblib

        joblib.dump(payload, MODEL_PATH)
    except Exception as exc:
        logger.warning(
            "joblib.dump failed (%s); falling back to pickle for %s", exc, MODEL_PATH
        )
        with open(MODEL_PATH, "wb") as fh:
            pickle.dump(payload, fh)

    # Write sidecar SHA-256 so load_best_model() can verify integrity.
    digest = _file_sha256(MODEL_PATH)
    with open(_SHA256_PATH, "w") as fh:
        fh.write(digest + "\n")

    return MODEL_PATH


def load_best_model() -> dict | None:
    """Load the persisted best model from disk.

    The SHA-256 checksum sidecar (MODEL_PATH + '.sha256') is always verified
    before any deserialization. If the checksum is absent or mismatches the
    file on disk the function returns None and logs an error instead of loading
    potentially tampered data.
    """
    if not os.path.exists(MODEL_PATH):
        return None

    # Checksum guard — must pass before any deserialization.
    if not os.path.exists(_SHA256_PATH):
        logger.error(
            "Refusing to load %s — checksum sidecar %s is missing. "
            "Re-train to regenerate a trusted model file.",
            MODEL_PATH,
            _SHA256_PATH,
        )
        return None

    with open(_SHA256_PATH) as fh:
        expected = fh.read().strip()
    actual = _file_sha256(MODEL_PATH)
    if actual != expected:
        logger.error(
            "SHA-256 mismatch for %s (expected %s, got %s) — "
            "refusing to deserialize potentially tampered file.",
            MODEL_PATH,
            expected,
            actual,
        )
        return None

    try:
        import joblib

        return joblib.load(MODEL_PATH)
    except Exception as exc:
        logger.warning(
            "joblib.load failed for %s: %s — trying pickle fallback", MODEL_PATH, exc
        )

    try:
        with open(MODEL_PATH, "rb") as fh:
            return pickle.load(fh)
    except Exception as exc:
        logger.warning("pickle.load also failed for %s: %s", MODEL_PATH, exc)
        return None


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class LocalAutoML:
    """
    Real AutoML pipeline for football outcome prediction.

    Steps
    -----
    1. Extract numeric feature columns from the dataset.
    2. Build target vector (target_home_win by default).
    3. Run k-fold cross-validation on each candidate model.
    4. Select best model by log-loss.
    5. Optionally refine with Optuna (if installed).
    6. Refit best model on the full dataset.
    7. Persist to disk and return rich result dict.

    Works without sklearn by falling back to pure-Python GaussianNB
    and SGD-LogisticRegression implementations.
    """

    K_FOLDS: int = 5

    def available_models(self) -> list[str]:
        models = ["gaussian_nb", "logistic_sgd"]
        if _has("sklearn"):
            models += [
                "sklearn_logistic",
                "sklearn_random_forest",
                "sklearn_gradient_boosting",
            ]
        if _has("xgboost"):
            models.append("xgboost")
        if _has("lightgbm"):
            models.append("lightgbm")
        if _has("catboost"):
            models.append("catboost")
        if _has("optuna"):
            models.append("optuna_tuned")
        return models

    def run(
        self,
        dataset_df,
        target_col: str = "target_home_win",
        feature_cols: list[str] | None = None,
        n_optuna_trials: int = 30,
    ) -> dict:
        """
        Parameters
        ----------
        dataset_df     : pd.DataFrame — output of DatasetBuilder.build_training_dataset()
        target_col     : binary target column (default 'target_home_win')
        feature_cols   : list of feature column names; auto-detected when None
        n_optuna_trials: number of Optuna trials (used only if optuna is installed)

        Returns
        -------
        dict with keys:
          status, available_models, best_model, best_log_loss, best_brier,
          best_accuracy, all_results, feature_importances, model_path, rows, features
        """
        available = self.available_models()
        base = {
            "available_models": available,
            "best_model": None,
            "status": "ok",
        }

        if dataset_df is None or (hasattr(dataset_df, "empty") and dataset_df.empty):
            return {**base, "status": "empty_dataset"}

        if target_col not in dataset_df.columns:
            return {**base, "status": "missing_target"}

        if feature_cols is None:
            exclude = {
                target_col,
                "target_draw",
                "target_away_win",
                "fixture_id",
                "league",
                "match_date",
                "home",
                "away",
                "home_goals",
                "away_goals",
                "season",
            }
            feature_cols = [
                c
                for c in dataset_df.columns
                if c not in exclude and dataset_df[c].dtype.kind in ("f", "i", "u")
            ]

        if not feature_cols:
            return {**base, "status": "no_numeric_features"}

        sub = dataset_df[feature_cols + [target_col]].dropna()
        if len(sub) < _MIN_ROWS:
            return {**base, "status": "insufficient_data", "rows": len(sub)}

        if sub[target_col].nunique() < 2:
            return {**base, "status": "single_class"}

        X_raw = sub[feature_cols].values.tolist()
        y_list = [int(v) for v in sub[target_col].tolist()]
        n_rows = len(X_raw)
        n_feats = len(feature_cols)

        all_results: dict[str, dict] = {}
        best_name: str = ""
        best_ll: float = float("inf")
        best_model: Any = None

        if _has("sklearn"):
            import numpy as np

            X_arr = np.array(X_raw)
            y_arr = np.array(y_list)

            for name, clf in _sklearn_candidates(n_rows):
                try:
                    metrics = _cv_sklearn(X_arr, y_arr, clf, k=self.K_FOLDS)
                    all_results[name] = metrics
                    if metrics["log_loss"] < best_ll:
                        best_ll = metrics["log_loss"]
                        best_name = name
                        best_model = clf
                except Exception as exc:
                    all_results[name] = {"error": str(exc)}

            if _has("optuna") and n_rows >= 50:
                try:
                    tuned_clf, tuned_ll = _optuna_tune(
                        X_raw, y_list, feature_cols, n_trials=n_optuna_trials
                    )
                    all_results["optuna_tuned"] = {"log_loss": tuned_ll}
                    if tuned_ll < best_ll:
                        best_ll = tuned_ll
                        best_name = "optuna_tuned"
                        best_model = tuned_clf
                except Exception as exc:
                    all_results["optuna_tuned"] = {"error": str(exc)}

            if best_model is not None:
                best_model.fit(X_arr, y_arr)

            feat_imp = _feature_importances_sklearn(
                best_model, feature_cols, X_arr, y_arr
            )

        else:
            pure_candidates = [
                ("gaussian_nb", lambda: _GaussianNB()),
                ("logistic_sgd", lambda: _SGDLogisticReg()),
            ]
            for name, factory in pure_candidates:
                try:
                    metrics = _cv_pure_python(X_raw, y_list, factory, k=self.K_FOLDS)
                    all_results[name] = metrics
                    if metrics["log_loss"] < best_ll:
                        best_ll = metrics["log_loss"]
                        best_name = name
                        best_model = factory()
                except Exception as exc:
                    all_results[name] = {"error": str(exc)}

            if best_model is not None:
                _, X_norm, *_ = _normalise(X_raw, X_raw)
                best_model.fit(X_norm, y_list)
                feat_imp = dict(
                    zip(
                        feature_cols,
                        best_model.feature_importances(n_feats),
                    )
                )
            else:
                feat_imp = {}

        meta = {
            "best_model_name": best_name,
            "feature_cols": feature_cols,
            "target_col": target_col,
            "best_log_loss": best_ll,
            "rows": n_rows,
        }
        model_path = None
        if best_model is not None:
            try:
                model_path = _save_model(best_model, meta)
            except Exception as exc:
                logger.warning("Model persistence failed for %s: %s", best_name, exc)

        return {
            "status": "trained",
            "available_models": available,
            "best_model": best_name,
            "best_log_loss": round(best_ll, 6),
            "best_brier": round(all_results.get(best_name, {}).get("brier", 1.0), 6),
            "best_accuracy": round(
                all_results.get(best_name, {}).get("accuracy", 0.0), 6
            ),
            "all_results": all_results,
            "feature_importances": feat_imp,
            "model_path": model_path,
            "rows": n_rows,
            "features": feature_cols,
        }

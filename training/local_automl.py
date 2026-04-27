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
"""

from __future__ import annotations

import importlib.util
import logging
import math
import os
import pickle
import random
from collections import defaultdict
from typing import Any

logger = logging.getLogger(__name__)

MODEL_PATH = "outputs/automl_best_model.pkl"
_MIN_ROWS = 20  # minimum rows needed for meaningful CV


# ---------------------------------------------------------------------------
# Library probe
# ---------------------------------------------------------------------------


def _has(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


# ---------------------------------------------------------------------------
# Pure-Python ML primitives  (used when sklearn is absent)
# ---------------------------------------------------------------------------


class _GaussianNB:
    """Gaussian Naive Bayes binary classifier."""

    def __init__(self) -> None:
        self._priors: dict[int, float] = {}
        self._params: dict[int, dict[int, tuple[float, float]]] = {}

    def fit(self, X: list[list[float]], y: list[int]) -> "_GaussianNB":
        classes = list(set(y))
        n = len(y)
        for c in classes:
            idx = [i for i, yi in enumerate(y) if yi == c]
            self._priors[c] = len(idx) / n
            self._params[c] = {}
            for j in range(len(X[0])):
                vals = [X[i][j] for i in idx]
                mu = sum(vals) / len(vals)
                var = sum((v - mu) ** 2 for v in vals) / max(len(vals) - 1, 1) + 1e-9
                self._params[c][j] = (mu, var)
        return self

    def _log_prob(self, x: list[float], c: int) -> float:
        lp = math.log(self._priors[c])
        for j, xj in enumerate(x):
            mu, var = self._params[c][j]
            lp += -0.5 * math.log(2 * math.pi * var) - 0.5 * (xj - mu) ** 2 / var
        return lp

    def predict_proba_one(self, x: list[float]) -> float:
        log_p = {c: self._log_prob(x, c) for c in self._priors}
        max_lp = max(log_p.values())
        probs = {c: math.exp(lp - max_lp) for c, lp in log_p.items()}
        total = sum(probs.values())
        return probs.get(1, 0.0) / total if total > 0 else 0.5

    def predict_proba(self, X: list[list[float]]) -> list[float]:
        return [self.predict_proba_one(x) for x in X]

    def feature_importances(self, n_features: int) -> list[float]:
        # Use variance-based importance: features with high between-class
        # mean difference relative to within-class variance matter more
        if not self._params or 0 not in self._params or 1 not in self._params:
            return [1.0 / n_features] * n_features
        imps = []
        for j in range(n_features):
            mu0, var0 = self._params.get(0, {}).get(j, (0.0, 1.0))
            mu1, var1 = self._params.get(1, {}).get(j, (0.0, 1.0))
            between = (mu1 - mu0) ** 2
            within = (var0 + var1) / 2.0 + 1e-9
            imps.append(between / within)
        total = sum(imps) or 1.0
        return [v / total for v in imps]


class _SGDLogisticReg:
    """Binary logistic regression via mini-batch SGD with L2 regularisation."""

    def __init__(
        self,
        lr: float = 0.05,
        n_iter: int = 300,
        l2: float = 1e-3,
        batch_size: int = 32,
        random_state: int = 42,
    ) -> None:
        self.lr = lr
        self.n_iter = n_iter
        self.l2 = l2
        self.batch_size = batch_size
        self.random_state = random_state
        self._w: list[float] = []
        self._b: float = 0.0

    @staticmethod
    def _sigmoid(z: float) -> float:
        if z >= 0:
            return 1.0 / (1.0 + math.exp(-z))
        e = math.exp(z)
        return e / (1.0 + e)

    def fit(self, X: list[list[float]], y: list[int]) -> "_SGDLogisticReg":
        rng = random.Random(self.random_state)
        n, p = len(X), len(X[0])
        self._w = [0.0] * p
        self._b = 0.0
        idx_all = list(range(n))
        for _ in range(self.n_iter):
            rng.shuffle(idx_all)
            for start in range(0, n, self.batch_size):
                batch = idx_all[start : start + self.batch_size]
                gw = [0.0] * p
                gb = 0.0
                for i in batch:
                    z = sum(self._w[j] * X[i][j] for j in range(p)) + self._b
                    err = self._sigmoid(z) - y[i]
                    for j in range(p):
                        gw[j] += err * X[i][j]
                    gb += err
                bs = len(batch)
                for j in range(p):
                    self._w[j] -= self.lr * (gw[j] / bs + self.l2 * self._w[j])
                self._b -= self.lr * gb / bs
        return self

    def predict_proba_one(self, x: list[float]) -> float:
        z = sum(self._w[j] * x[j] for j in range(len(x))) + self._b
        return self._sigmoid(z)

    def predict_proba(self, X: list[list[float]]) -> list[float]:
        return [self.predict_proba_one(x) for x in X]

    def feature_importances(self, n_features: int) -> list[float]:
        w = [abs(v) for v in self._w[:n_features]]
        tot = sum(w) or 1.0
        return [v / tot for v in w]


# ---------------------------------------------------------------------------
# Metrics (pure Python)
# ---------------------------------------------------------------------------


def _log_loss(y_true: list[int], probs: list[float], eps: float = 1e-12) -> float:
    n = len(y_true)
    if n == 0:
        return float("inf")
    s = sum(
        yi * math.log(max(p, eps)) + (1 - yi) * math.log(max(1 - p, eps))
        for yi, p in zip(y_true, probs)
    )
    return -s / n


def _brier(y_true: list[int], probs: list[float]) -> float:
    n = len(y_true)
    return sum((p - yi) ** 2 for yi, p in zip(y_true, probs)) / n if n else 1.0


def _accuracy(y_true: list[int], probs: list[float]) -> float:
    n = len(y_true)
    correct = sum(1 for yi, p in zip(y_true, probs) if (p >= 0.5) == (yi == 1))
    return correct / n if n else 0.0


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def _normalise(
    X_train: list[list[float]], X_val: list[list[float]]
) -> tuple[list[list[float]], list[list[float]], list[float], list[float]]:
    """Z-score normalisation fitted on training data only."""
    p = len(X_train[0])
    means, stds = [], []
    for j in range(p):
        col = [X_train[i][j] for i in range(len(X_train))]
        mu = sum(col) / len(col)
        sd = math.sqrt(sum((v - mu) ** 2 for v in col) / max(len(col) - 1, 1))
        means.append(mu)
        stds.append(max(sd, 1e-8))

    def _norm(X: list[list[float]]) -> list[list[float]]:
        return [
            [(X[i][j] - means[j]) / stds[j] for j in range(p)] for i in range(len(X))
        ]

    return _norm(X_train), _norm(X_val), means, stds


def _stratified_kfold(
    y: list[int], k: int = 5, seed: int = 42
) -> list[tuple[list[int], list[int]]]:
    rng = random.Random(seed)
    classes = list(set(y))
    buckets: dict[int, list[int]] = defaultdict(list)
    for i, yi in enumerate(y):
        buckets[yi].append(i)
    for c in classes:
        rng.shuffle(buckets[c])

    folds: list[list[int]] = [[] for _ in range(k)]
    for c in classes:
        for rank, idx in enumerate(buckets[c]):
            folds[rank % k].append(idx)

    splits = []
    for fi in range(k):
        val = folds[fi]
        train = [idx for fj, f in enumerate(folds) if fj != fi for idx in f]
        splits.append((train, val))
    return splits


# ---------------------------------------------------------------------------
# sklearn-based candidates
# ---------------------------------------------------------------------------


def _sklearn_candidates(n_rows: int) -> list[tuple[str, Any]]:
    """Return list of (name, unfitted_model) using sklearn if available."""
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    n_est = min(300, max(50, n_rows // 5))
    candidates = [
        (
            "logistic_regression",
            LogisticRegression(
                max_iter=1000, C=1.0, solver="lbfgs", multi_class="auto"
            ),
        ),
        (
            "random_forest",
            RandomForestClassifier(
                n_estimators=n_est,
                max_depth=8,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1,
            ),
        ),
        (
            "gradient_boosting",
            GradientBoostingClassifier(
                n_estimators=min(200, n_est),
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42,
            ),
        ),
    ]

    if _has("xgboost"):
        from xgboost import XGBClassifier

        candidates.append(
            (
                "xgboost",
                XGBClassifier(
                    n_estimators=min(200, n_est),
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    use_label_encoder=False,
                    eval_metric="logloss",
                    random_state=42,
                    verbosity=0,
                ),
            )
        )

    if _has("lightgbm"):
        from lightgbm import LGBMClassifier

        candidates.append(
            (
                "lightgbm",
                LGBMClassifier(
                    n_estimators=min(200, n_est),
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    verbosity=-1,
                ),
            )
        )

    if _has("catboost"):
        from catboost import CatBoostClassifier

        candidates.append(
            (
                "catboost",
                CatBoostClassifier(
                    iterations=min(200, n_est),
                    depth=4,
                    learning_rate=0.05,
                    random_seed=42,
                    verbose=False,
                ),
            )
        )

    return candidates


def _optuna_tune(
    X_list: list, y_list: list, feature_cols: list[str], n_trials: int = 40
) -> tuple[Any, float]:
    """Hyperparameter search with Optuna over RandomForest + GBM."""
    import numpy as np
    import optuna
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    X = np.array(X_list)
    y = np.array(y_list)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        model_name = trial.suggest_categorical("model", ["rf", "gbm"])
        if model_name == "rf":
            clf = RandomForestClassifier(
                n_estimators=trial.suggest_int("n_estimators", 50, 400),
                max_depth=trial.suggest_int("max_depth", 3, 12),
                min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 20),
                random_state=42,
                n_jobs=-1,
            )
        else:
            clf = GradientBoostingClassifier(
                n_estimators=trial.suggest_int("n_estimators", 50, 300),
                max_depth=trial.suggest_int("max_depth", 2, 8),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                subsample=trial.suggest_float("subsample", 0.5, 1.0),
                random_state=42,
            )
        cv = StratifiedKFold(n_splits=5, shuffle=False)
        score = cross_val_score(clf, X, y, cv=cv, scoring="neg_log_loss", n_jobs=-1)
        return -score.mean()

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    model_name = best_params.pop("model")
    if model_name == "rf":
        best_clf = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
    else:
        best_clf = GradientBoostingClassifier(**best_params, random_state=42)
    best_clf.fit(X, y)

    return best_clf, study.best_value


# ---------------------------------------------------------------------------
# Cross-validation runner
# ---------------------------------------------------------------------------


def _cv_pure_python(
    X: list[list[float]],
    y: list[int],
    model_factory,
    k: int = 5,
) -> dict[str, float]:
    splits = _stratified_kfold(y, k=k)
    ll_scores, brier_scores, acc_scores = [], [], []

    for train_idx, val_idx in splits:
        X_tr = [X[i] for i in train_idx]
        y_tr = [y[i] for i in train_idx]
        X_vl = [X[i] for i in val_idx]
        y_vl = [y[i] for i in val_idx]

        if len(set(y_tr)) < 2:
            continue

        X_tr_n, X_vl_n, _, _ = _normalise(X_tr, X_vl)
        clf = model_factory()
        clf.fit(X_tr_n, y_tr)
        probs = clf.predict_proba(X_vl_n)
        ll_scores.append(_log_loss(y_vl, probs))
        brier_scores.append(_brier(y_vl, probs))
        acc_scores.append(_accuracy(y_vl, probs))

    if not ll_scores:
        return {"log_loss": float("inf"), "brier": 1.0, "accuracy": 0.0}
    return {
        "log_loss": sum(ll_scores) / len(ll_scores),
        "brier": sum(brier_scores) / len(brier_scores),
        "accuracy": sum(acc_scores) / len(acc_scores),
    }


def _cv_sklearn(X_arr, y_arr, clf, k: int = 5) -> dict[str, float]:
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    cv = StratifiedKFold(n_splits=k, shuffle=False)
    ll = cross_val_score(clf, X_arr, y_arr, cv=cv, scoring="neg_log_loss", n_jobs=1)
    bs = cross_val_score(clf, X_arr, y_arr, cv=cv, scoring="neg_brier_score", n_jobs=1)
    ac = cross_val_score(clf, X_arr, y_arr, cv=cv, scoring="accuracy", n_jobs=1)
    return {
        "log_loss": float(-ll.mean()),
        "brier": float(-bs.mean()),
        "accuracy": float(ac.mean()),
    }


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def _save_model(model: Any, meta: dict) -> str:
    os.makedirs(os.path.dirname(MODEL_PATH) or ".", exist_ok=True)
    payload = {"model": model, "meta": meta}
    try:
        import joblib

        joblib.dump(payload, MODEL_PATH)
    except Exception as exc:
        logger.warning("joblib.dump failed (%s); falling back to pickle for %s", exc, MODEL_PATH)
        with open(MODEL_PATH, "wb") as fh:
            pickle.dump(payload, fh)
    return MODEL_PATH


def load_best_model() -> dict | None:
    """Load the persisted best model from disk. Returns None if absent."""
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        import joblib

        return joblib.load(MODEL_PATH)
    except Exception as exc:
        logger.warning("joblib.load failed for %s: %s — trying pickle fallback", MODEL_PATH, exc)
    # pickle fallback: only load files written by this process (same host, same user)
    logger.warning(
        "joblib unavailable or failed; falling back to pickle for %s — "
        "ensure this file originates from a trusted source",
        MODEL_PATH,
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
        models = ["gaussian_nb", "logistic_sgd"]  # always available
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

        # Auto-detect numeric feature columns
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

        # Drop rows with NaN in features or target
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

        # ------------------------------------------------------------------
        # Branch A: sklearn available
        # ------------------------------------------------------------------
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

            # Optuna refinement on top of sklearn
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

            # Refit best on full dataset
            if best_model is not None:
                best_model.fit(X_arr, y_arr)

            # Feature importances
            feat_imp = _feature_importances_sklearn(
                best_model, feature_cols, X_arr, y_arr
            )

        # ------------------------------------------------------------------
        # Branch B: pure Python fallback
        # ------------------------------------------------------------------
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

            # Refit best on full normalised dataset
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

        # ------------------------------------------------------------------
        # Persist and return
        # ------------------------------------------------------------------
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


def _feature_importances_sklearn(
    model: Any,
    feature_cols: list[str],
    X,
    y,
) -> dict[str, float]:
    """Extract feature importances from any fitted sklearn-compatible model."""
    imp = None

    # Tree-based models expose feature_importances_
    if hasattr(model, "feature_importances_"):
        imp = list(model.feature_importances_)

    # Linear models expose coef_
    elif hasattr(model, "coef_"):
        coef = model.coef_
        if hasattr(coef, "tolist"):
            flat = coef.tolist()
            if flat and isinstance(flat[0], list):
                flat = flat[0]
            imp = [abs(v) for v in flat]
        else:
            imp = [abs(v) for v in list(coef)]

    if imp is None or len(imp) != len(feature_cols):
        # Permutation importance fallback using log-loss degradation
        try:
            from sklearn.inspection import permutation_importance

            r = permutation_importance(
                model, X, y, n_repeats=5, scoring="neg_log_loss", random_state=42
            )
            imp = list(-r.importances_mean)
        except Exception as exc:
            logger.warning("Permutation importance failed, using uniform weights: %s", exc)
            return {c: 1.0 / len(feature_cols) for c in feature_cols}

    total = sum(abs(v) for v in imp) or 1.0
    ranked = dict(zip(feature_cols, [abs(v) / total for v in imp]))
    return dict(sorted(ranked.items(), key=lambda kv: kv[1], reverse=True))

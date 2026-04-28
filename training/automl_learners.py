"""
Pure-Python and sklearn ML learner implementations for AutoML.

Provides _GaussianNB and _SGDLogisticReg as fallbacks when sklearn is absent,
plus _sklearn_candidates() and _optuna_tune() for full-library environments.
"""

from __future__ import annotations

import importlib.util
import math
import random
from typing import Any


def _has(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


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

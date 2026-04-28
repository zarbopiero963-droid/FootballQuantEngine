"""
Metrics, data helpers, cross-validation runners, and feature importance
extraction for AutoML.  Pure-Python where possible; sklearn used only where
explicitly required (CV runners and permutation importance).
"""

from __future__ import annotations

import logging
import math
import random
from collections import defaultdict
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metrics
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
# Cross-validation runners
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
# Feature importance extraction
# ---------------------------------------------------------------------------


def _feature_importances_sklearn(
    model: Any,
    feature_cols: list[str],
    X,
    y,
) -> dict[str, float]:
    """Extract feature importances from any fitted sklearn-compatible model."""
    imp = None

    if hasattr(model, "feature_importances_"):
        imp = list(model.feature_importances_)
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
        try:
            from sklearn.inspection import permutation_importance

            r = permutation_importance(
                model, X, y, n_repeats=5, scoring="neg_log_loss", random_state=42
            )
            imp = list(-r.importances_mean)
        except Exception as exc:
            logger.warning(
                "Permutation importance failed, using uniform weights: %s", exc
            )
            return {c: 1.0 / len(feature_cols) for c in feature_cols}

    total = sum(abs(v) for v in imp) or 1.0
    ranked = dict(zip(feature_cols, [abs(v) / total for v in imp]))
    return dict(sorted(ranked.items(), key=lambda kv: kv[1], reverse=True))

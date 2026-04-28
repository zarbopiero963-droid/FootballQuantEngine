"""
ML ensemble model for 3-way football outcome prediction.

Combines up to three classifiers in a soft-voting ensemble:
  1. LogisticRegression (sklearn — always available)
  2. XGBClassifier     (xgboost — used if installed)
  3. LGBMClassifier    (lightgbm — used if installed)

Each classifier produces a probability vector [p_home, p_draw, p_away].
The ensemble averages these vectors (equal weight per available model) and
applies per-outcome Isotonic Regression calibration on held-out data.

Temporal train/calibration split
---------------------------------
fit() accepts a list of match dicts (chronologically ordered).  The last
20% of rows are reserved as a calibration set (no shuffle — no look-ahead).
Isotonic calibration is fitted on this held-out slice.

Feature vector (11 dimensions)
--------------------------------
  elo_diff, form_diff, xg_diff, xga_diff, xpts_diff,
  shots_diff, shots_on_target_diff, form_score_diff,
  bookmaker_odds_home, bookmaker_odds_draw, bookmaker_odds_away

Target encoding
---------------
  H=0, D=1, A=2  (home win / draw / away win)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional imports — all gracefully absent
# ---------------------------------------------------------------------------

try:
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    _SKLEARN = True
except Exception:
    LogisticRegression = None
    IsotonicRegression = None
    StandardScaler = None
    _SKLEARN = False

try:
    from xgboost import XGBClassifier

    _XGB = True
except Exception:
    XGBClassifier = None
    _XGB = False

try:
    from lightgbm import LGBMClassifier

    _LGBM = True
except Exception:
    LGBMClassifier = None
    _LGBM = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

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

_OUTCOME_MAP = {"H": 0, "D": 1, "A": 2}
_CALIBRATION_FRAC = 0.20
_MIN_SAMPLES = 30


# ---------------------------------------------------------------------------
# MLEnsemble
# ---------------------------------------------------------------------------


class MLEnsemble:
    """
    Soft-voting ensemble of LR + XGBoost + LightGBM for 3-way prediction.

    Usage
    -----
        ensemble = MLEnsemble()
        ensemble.fit(train_rows)                  # list of dicts with result + features
        probs = ensemble.predict_proba(row_dict)  # {"home": p, "draw": p, "away": p}
    """

    def __init__(self) -> None:
        self._classifiers: List = []
        self._calibrators: List[Optional[object]] = []  # one per outcome (H/D/A)
        self._scaler = None
        self.is_fitted = False
        self.n_models = 0
        self.n_train = 0
        self.feature_importances_: Optional[Dict[str, float]] = None

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, rows: List[Dict]) -> "MLEnsemble":
        """
        Fit ensemble on a chronologically-ordered list of match dicts.

        Each dict must contain:
          - ``result``: "H", "D", or "A"
          - One key per FEATURE_COLUMNS (missing → 0.0)

        Returns self.
        """
        self._classifiers = []
        self._calibrators = [None, None, None]
        self.is_fitted = False
        self.n_models = 0

        if not rows or not _SKLEARN:
            logger.warning("MLEnsemble.fit: sklearn unavailable or no data")
            return self

        X, y = _build_arrays(rows)
        if X is None or len(X) < _MIN_SAMPLES:
            logger.warning(
                "MLEnsemble.fit: only %d samples, need %d",
                len(X) if X is not None else 0,
                _MIN_SAMPLES,
            )
            return self

        # Temporal split: last 20% for calibration
        n = len(X)
        split = max(int(n * (1 - _CALIBRATION_FRAC)), _MIN_SAMPLES)
        X_train, y_train = X[:split], y[:split]
        X_cal, y_cal = X[split:], y[split:]

        # Feature scaling
        self._scaler = StandardScaler()
        X_train_s = self._scaler.fit_transform(X_train)
        X_cal_s = self._scaler.transform(X_cal) if len(X_cal) > 0 else X_cal

        # Build and fit each classifier
        candidates = _build_classifiers()
        fitted = []
        for name, clf in candidates:
            try:
                clf.fit(X_train_s, y_train)
                fitted.append((name, clf))
                logger.info("MLEnsemble: fitted %s on %d samples", name, len(X_train))
            except Exception as exc:  # noqa: BLE001
                logger.warning("MLEnsemble: %s failed to fit — %s", name, exc)

        if not fitted:
            logger.warning("MLEnsemble: no classifiers could be fitted")
            return self

        self._classifiers = [clf for _, clf in fitted]
        self.n_models = len(self._classifiers)
        self.n_train = split

        # Fit Isotonic calibrators on held-out set (one per outcome)
        if len(X_cal) >= 10:
            raw_probs = self._raw_ensemble_proba(X_cal_s)
            for outcome_idx in range(3):
                y_binary = (y_cal == outcome_idx).astype(float)
                p_raw = raw_probs[:, outcome_idx]
                try:
                    cal = IsotonicRegression(out_of_bounds="clip")
                    cal.fit(p_raw, y_binary)
                    self._calibrators[outcome_idx] = cal
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "MLEnsemble: calibration failed for outcome %d — %s",
                        outcome_idx,
                        exc,
                    )

        # Feature importance (from first tree-based model if available)
        self.feature_importances_ = _extract_importances(fitted)

        self.is_fitted = True
        logger.info(
            "MLEnsemble: ready — %d models, train=%d cal=%d",
            self.n_models,
            split,
            len(X_cal),
        )
        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_proba(self, row: Dict) -> Dict[str, float]:
        """
        Return calibrated 3-way probabilities for one match dict.

        Returns {"home": p_H, "draw": p_D, "away": p_A} summing to 1.0.
        Falls back to uniform {1/3, 1/3, 1/3} if not fitted.
        """
        if not self.is_fitted or not self._classifiers:
            return {"home": 1 / 3, "draw": 1 / 3, "away": 1 / 3}

        x = _row_to_array(row)
        x_s = self._scaler.transform(x.reshape(1, -1))

        raw = self._raw_ensemble_proba(x_s)[0]  # shape (3,)

        # Apply Isotonic calibration per outcome, then renormalise
        calibrated = np.array(
            [
                (
                    float(self._calibrators[i].predict([raw[i]])[0])
                    if self._calibrators[i] is not None
                    else float(raw[i])
                )
                for i in range(3)
            ]
        )
        calibrated = np.clip(calibrated, 1e-6, 1.0)
        calibrated /= calibrated.sum()

        return {
            "home": float(calibrated[0]),
            "draw": float(calibrated[1]),
            "away": float(calibrated[2]),
        }

    def predict_home_probability(self, row: Dict) -> float:
        """Convenience alias for backward compatibility with MLLayerReal callers."""
        return self.predict_proba(row)["home"]

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def summary(self) -> str:
        if not self.is_fitted:
            return "MLEnsemble(not fitted)"
        names = []
        for clf in self._classifiers:
            names.append(type(clf).__name__)
        return (
            f"MLEnsemble(models=[{', '.join(names)}], "
            f"n_train={self.n_train}, calibrated={sum(c is not None for c in self._calibrators)}/3)"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _raw_ensemble_proba(self, X_scaled: np.ndarray) -> np.ndarray:
        """Average predict_proba across all fitted classifiers."""
        proba_sum = np.zeros((len(X_scaled), 3))
        for clf in self._classifiers:
            p = clf.predict_proba(X_scaled)
            if p.shape[1] == 3:
                proba_sum += p
            elif p.shape[1] == 2:
                # Binary classifier — fill home (idx 0) only
                proba_sum[:, 0] += p[:, 1]
        return proba_sum / max(len(self._classifiers), 1)


# ---------------------------------------------------------------------------
# Module helpers
# ---------------------------------------------------------------------------


def _build_classifiers() -> List[Tuple[str, object]]:
    """Return list of (name, unfitted_classifier) for available libraries."""
    candidates = []

    if _SKLEARN and LogisticRegression is not None:
        candidates.append(
            (
                "LogisticRegression",
                LogisticRegression(
                    max_iter=2000,
                    C=1.0,
                    multi_class="multinomial",
                    solver="lbfgs",
                    random_state=42,
                ),
            )
        )

    if _XGB and XGBClassifier is not None:
        candidates.append(
            (
                "XGBClassifier",
                XGBClassifier(
                    n_estimators=300,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    use_label_encoder=False,
                    eval_metric="mlogloss",
                    random_state=42,
                    verbosity=0,
                    num_class=3,
                    objective="multi:softprob",
                ),
            )
        )

    if _LGBM and LGBMClassifier is not None:
        candidates.append(
            (
                "LGBMClassifier",
                LGBMClassifier(
                    n_estimators=300,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    num_class=3,
                    objective="multiclass",
                    random_state=42,
                    verbose=-1,
                ),
            )
        )

    return candidates


def _build_arrays(
    rows: List[Dict],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Extract (X, y) arrays from list of match dicts."""
    X_list, y_list = [], []
    for r in rows:
        result = str(r.get("result", "")).strip().upper()
        if result not in _OUTCOME_MAP:
            continue
        y_list.append(_OUTCOME_MAP[result])
        X_list.append(_row_to_array(r))

    if not X_list:
        return None, None

    return np.array(X_list, dtype=float), np.array(y_list, dtype=int)


def _row_to_array(row: Dict) -> np.ndarray:
    return np.array(
        [float(row.get(col, 0.0) or 0.0) for col in FEATURE_COLUMNS],
        dtype=float,
    )


def _extract_importances(
    fitted: List[Tuple[str, object]],
) -> Optional[Dict[str, float]]:
    """Extract feature importances from the first tree-based model."""
    for name, clf in fitted:
        if hasattr(clf, "feature_importances_"):
            imp = clf.feature_importances_
            return {col: float(imp[i]) for i, col in enumerate(FEATURE_COLUMNS)}
    return None

from __future__ import annotations

import numpy as np

try:
    from sklearn.isotonic import IsotonicRegression
except Exception:
    IsotonicRegression = None


class ProbabilityCalibratorReal:

    def __init__(self):
        self.model = None
        self.is_fitted = False

    def fit(self, raw_probs, y_true):
        raw_probs = np.asarray(raw_probs, dtype=float)
        y_true = np.asarray(y_true, dtype=float)

        if len(raw_probs) == 0 or len(y_true) == 0 or len(raw_probs) != len(y_true):
            self.is_fitted = False
            self.model = None
            return

        if IsotonicRegression is None:
            self.is_fitted = False
            self.model = None
            return

        self.model = IsotonicRegression(out_of_bounds="clip")
        self.model.fit(raw_probs, y_true)
        self.is_fitted = True

    def transform(self, raw_probs):
        raw_probs = np.asarray(raw_probs, dtype=float)

        if not self.is_fitted or self.model is None:
            return np.clip(raw_probs, 0.0001, 0.9999)

        calibrated = self.model.predict(raw_probs)
        return np.clip(calibrated, 0.0001, 0.9999)

    def transform_one(self, raw_prob: float) -> float:
        return float(self.transform([raw_prob])[0])

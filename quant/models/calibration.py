from __future__ import annotations

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


class ProbabilityCalibration:
    """
    Three-way probability calibration.

    By default uses 8% linear shrinkage toward uniform (original behaviour).
    After calling fit() with labelled validation data, switches to Platt
    scaling (logistic regression) or isotonic regression, whichever scores
    lower log-loss on the fitted data.  Falls back to linear shrinkage if
    sklearn is unavailable.
    """

    def __init__(self, shrink=0.08):
        self.shrink = shrink
        self._platt: Optional[object] = None    # fitted sklearn calibrator
        self._iso: Optional[object] = None
        self._use_platt: bool = False

    def calibrate_three_way(self, probs):
        home = float(probs.get("home_win", 0.0))
        draw = float(probs.get("draw", 0.0))
        away = float(probs.get("away_win", 0.0))

        total = home + draw + away
        if total <= 0:
            return {
                "home_win": 1 / 3,
                "draw": 1 / 3,
                "away_win": 1 / 3,
            }

        home /= total
        draw /= total
        away /= total

        uniform = 1 / 3

        home = (1 - self.shrink) * home + self.shrink * uniform
        draw = (1 - self.shrink) * draw + self.shrink * uniform
        away = (1 - self.shrink) * away + self.shrink * uniform

        total = home + draw + away

        return {
            "home_win": home / total,
            "draw": draw / total,
            "away_win": away / total,
        }

    def fit(self, predicted_probs: List[float], outcomes: List[float]) -> None:
        """
        Fit Platt scaling (logistic regression) and isotonic regression on
        validation predictions vs actual binary outcomes (1.0 = positive).

        The better method (lower log-loss) is stored and used automatically
        in subsequent calibrate_binary() calls.  Three-way calibrate_three_way()
        is not affected — call fit() separately for each outcome dimension if
        needed.

        Parameters
        ----------
        predicted_probs : list of floats in [0, 1]
        outcomes        : list of floats — 0.0, 0.5 (draw treated as 0), or 1.0
        """
        try:
            import numpy as np
            from sklearn.isotonic import IsotonicRegression
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import log_loss
        except ImportError:
            logger.warning("sklearn not available — ProbabilityCalibration.fit() skipped, using shrinkage")
            return

        if len(predicted_probs) < 10:
            logger.warning("Too few samples (%d) for calibration fit; using shrinkage", len(predicted_probs))
            return

        X = np.array(predicted_probs, dtype=float).reshape(-1, 1)
        y = np.array(outcomes, dtype=float)

        # Platt scaling: logistic regression on raw model scores
        platt = LogisticRegression(C=1e6)
        try:
            platt.fit(X, y)
            platt_loss = log_loss(y, platt.predict_proba(X)[:, 1])
        except Exception as exc:
            logger.warning("Platt fit failed: %s", exc)
            platt_loss = float("inf")

        # Isotonic regression
        iso = IsotonicRegression(out_of_bounds="clip")
        try:
            iso.fit(X.ravel(), y)
            iso_preds = np.clip(iso.predict(X.ravel()), 1e-7, 1 - 1e-7)
            iso_loss = log_loss(y, iso_preds)
        except Exception as exc:
            logger.warning("Isotonic fit failed: %s", exc)
            iso_loss = float("inf")

        if platt_loss <= iso_loss and platt_loss < float("inf"):
            self._platt = platt
            self._use_platt = True
            logger.info("ProbabilityCalibration: using Platt scaling (log-loss=%.5f)", platt_loss)
        elif iso_loss < float("inf"):
            self._iso = iso
            self._use_platt = False
            logger.info("ProbabilityCalibration: using isotonic regression (log-loss=%.5f)", iso_loss)
        else:
            logger.warning("Both calibrators failed; falling back to linear shrinkage")

    def calibrate_binary(self, p):
        p = float(p)
        p = max(0.0001, min(0.9999, p))

        if self._use_platt and self._platt is not None:
            try:
                import numpy as np
                prob = self._platt.predict_proba(np.array([[p]]))[0, 1]
                return float(np.clip(prob, 0.0001, 0.9999))
            except Exception:
                pass
        elif not self._use_platt and self._iso is not None:
            try:
                import numpy as np
                prob = self._iso.predict(np.array([p]))[0]
                return float(np.clip(prob, 0.0001, 0.9999))
            except Exception:
                pass

        return (1 - self.shrink) * p + self.shrink * 0.5

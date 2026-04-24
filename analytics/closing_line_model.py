"""
Closing Line Value (CLV) prediction model.

Closing Line Value measures how well a bettor's odds compare to the market
consensus at match kick-off.  A positive CLV means the bet was placed at
better odds than closing — a robust long-run profitability signal.

Model architecture
------------------
1.  **OddsRegressor** — fits a Ridge-regularised linear regression that maps
    opening odds (log-space) to closing odds.  Trained on historical fixture
    data stored in the local database.

2.  **ClosingLineModel** — wraps the regressor, exposes:
    - ``fit(records)`` — train/update from a list of historical dicts
    - ``predict_closing(opening_odds)`` — predict expected closing odds for 1×2
    - ``clv(bet_odds, predicted_closing)`` — log CLV in percentage points
    - ``score_bets(bets)`` — batch-score a list of current bet dicts

Mathematics
-----------
- Log-odds transform: x = ln(odds)  (ensures positivity after inverse)
- CLV = 100 × [ln(bet_odds) − ln(closing_odds)]
  positive CLV → bet odds better than closing (value bet confirmed)
- Predicted closing odds: exp(w @ x_opening + b)
- Regularisation: Ridge (L2, α=0.5) avoids overfitting on sparse league data

Usage
-----
    from analytics.closing_line_model import ClosingLineModel

    model = ClosingLineModel()
    model.fit(historical_records)           # list[dict] from DB
    scores = model.score_bets(today_bets)   # list[dict] with clv_pct added
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Numerics — numpy optional
# ---------------------------------------------------------------------------

try:
    import numpy as np

    _NP = True
except ImportError:
    _NP = False


# ---------------------------------------------------------------------------
# Pure-Python Ridge regression (fallback when numpy absent)
# ---------------------------------------------------------------------------


class _RidgePy:
    """Minimal Ridge regression implemented in pure Python."""

    def __init__(self, alpha: float = 0.5) -> None:
        self.alpha = alpha
        self.coef_: List[float] = []
        self.intercept_: float = 0.0
        self._fitted = False

    def fit(self, X: List[List[float]], y: List[float]) -> "_RidgePy":
        n = len(X)
        if n == 0:
            return self
        p = len(X[0])
        # Normal equations: (X^T X + αI) w = X^T y
        # Build X^T X  (p×p)
        XtX = [[0.0] * p for _ in range(p)]
        Xty = [0.0] * p
        for row, yi in zip(X, y):
            for i in range(p):
                Xty[i] += row[i] * yi
                for j in range(p):
                    XtX[i][j] += row[i] * row[j]
        for i in range(p):
            XtX[i][i] += self.alpha
        # Gaussian elimination
        A = [row[:] + [Xty[i]] for i, row in enumerate(XtX)]
        for col in range(p):
            pivot = max(range(col, p), key=lambda r: abs(A[r][col]))
            A[col], A[pivot] = A[pivot], A[col]
            if abs(A[col][col]) < 1e-12:
                continue
            for row in range(p):
                if row != col:
                    factor = A[row][col] / A[col][col]
                    for k in range(col, p + 1):
                        A[row][k] -= factor * A[col][k]
        self.coef_ = [
            A[i][p] / A[i][i] if abs(A[i][i]) > 1e-12 else 0.0 for i in range(p)
        ]
        self.intercept_ = 0.0
        self._fitted = True
        return self

    def predict(self, X: List[List[float]]) -> List[float]:
        return [
            sum(w * xi for w, xi in zip(self.coef_, row)) + self.intercept_ for row in X
        ]


# ---------------------------------------------------------------------------
# Numpy-backed Ridge (preferred when numpy available)
# ---------------------------------------------------------------------------


class _RidgeNp:
    """Ridge regression using numpy for efficiency."""

    def __init__(self, alpha: float = 0.5) -> None:
        self.alpha = alpha
        self.coef_: "np.ndarray" = np.zeros(1)
        self.intercept_: float = 0.0
        self._fitted = False

    def fit(self, X: List[List[float]], y: List[float]) -> "_RidgeNp":
        Xa = np.array(X, dtype=float)
        ya = np.array(y, dtype=float)
        n, p = Xa.shape
        A = Xa.T @ Xa + self.alpha * np.eye(p)
        b = Xa.T @ ya
        self.coef_ = np.linalg.solve(A, b)
        self.intercept_ = 0.0
        self._fitted = True
        return self

    def predict(self, X: List[List[float]]) -> List[float]:
        return list(np.array(X, dtype=float) @ self.coef_ + self.intercept_)


def _make_ridge(alpha: float = 0.5):
    return _RidgeNp(alpha) if _NP else _RidgePy(alpha)


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------


def _log_odds(o: float) -> float:
    """Safe log-odds transform (clamp odds to [1.01, 100])."""
    return math.log(max(1.01, min(o, 100.0)))


def _opening_features(home_o: float, draw_o: float, away_o: float) -> List[float]:
    """
    6-feature vector from opening 1×2 odds:
      [ln(home), ln(draw), ln(away),
       ln(home)*ln(draw), ln(home)*ln(away), ln(draw)*ln(away)]
    Quadratic interactions capture non-linear market dynamics.
    """
    lh, ld, la = _log_odds(home_o), _log_odds(draw_o), _log_odds(away_o)
    return [lh, ld, la, lh * ld, lh * la, ld * la]


# ---------------------------------------------------------------------------
# Per-market regressor bundle
# ---------------------------------------------------------------------------


@dataclass
class _MarketRegressors:
    """Three independent Ridge regressors for home/draw/away log-closing-odds."""

    home: object = field(default_factory=lambda: _make_ridge())
    draw: object = field(default_factory=lambda: _make_ridge())
    away: object = field(default_factory=lambda: _make_ridge())
    n_train: int = 0


# ---------------------------------------------------------------------------
# CLV result
# ---------------------------------------------------------------------------


@dataclass
class CLVResult:
    """Per-bet CLV assessment."""

    fixture_id: str
    market: str  # "home" | "draw" | "away"
    bet_odds: float
    predicted_closing: float
    clv_pct: float  # positive = value confirmed
    clv_grade: str  # "EXCELLENT" / "GOOD" / "NEUTRAL" / "POOR"
    confidence: float  # 0–1, derived from training sample size

    def as_dict(self) -> Dict:
        return {
            "fixture_id": self.fixture_id,
            "market": self.market,
            "bet_odds": round(self.bet_odds, 4),
            "predicted_closing": round(self.predicted_closing, 4),
            "clv_pct": round(self.clv_pct, 4),
            "clv_grade": self.clv_grade,
            "clv_confidence": round(self.confidence, 4),
        }


def _clv_grade(clv_pct: float) -> str:
    if clv_pct >= 3.0:
        return "EXCELLENT"
    if clv_pct >= 1.0:
        return "GOOD"
    if clv_pct >= -1.0:
        return "NEUTRAL"
    return "POOR"


def _model_confidence(n_train: int) -> float:
    """Sigmoid confidence: 0 at n=0, ~0.95 at n=500."""
    return 1.0 / (1.0 + math.exp(-0.012 * (n_train - 200))) if n_train > 0 else 0.0


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class ClosingLineModel:
    """
    CLV prediction model trained on historical opening→closing odds pairs.

    Parameters
    ----------
    alpha        : Ridge regularisation strength (default 0.5)
    min_samples  : minimum training rows before predictions are made (default 30)
    league_split : if True, fit separate models per league (default True)
    """

    def __init__(
        self,
        alpha: float = 0.5,
        min_samples: int = 30,
        league_split: bool = True,
    ) -> None:
        self._alpha = alpha
        self._min_samples = min_samples
        self._league_split = league_split
        # league → _MarketRegressors  (or "_global" key for pooled)
        self._regressors: Dict[str, _MarketRegressors] = {}
        self._global_reg: _MarketRegressors = _MarketRegressors()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, records: List[Dict]) -> "ClosingLineModel":
        """
        Train on historical records.

        Each record must contain:
          opening_home_odds, opening_draw_odds, opening_away_odds
          closing_home_odds, closing_draw_odds, closing_away_odds
        Optional:
          league (str)
        """
        if not records:
            return self

        # Partition by league
        by_league: Dict[str, List[Dict]] = {}
        global_X: List[List[float]] = []
        global_yh: List[float] = []
        global_yd: List[float] = []
        global_ya: List[float] = []

        for r in records:
            try:
                feats = _opening_features(
                    float(r["opening_home_odds"]),
                    float(r["opening_draw_odds"]),
                    float(r["opening_away_odds"]),
                )
                yh = _log_odds(float(r["closing_home_odds"]))
                yd = _log_odds(float(r["closing_draw_odds"]))
                ya = _log_odds(float(r["closing_away_odds"]))
            except (KeyError, TypeError, ValueError):
                continue

            global_X.append(feats)
            global_yh.append(yh)
            global_yd.append(yd)
            global_ya.append(ya)

            league = str(r.get("league", "Unknown"))
            if league not in by_league:
                by_league[league] = {"X": [], "yh": [], "yd": [], "ya": []}
            by_league[league]["X"].append(feats)
            by_league[league]["yh"].append(yh)
            by_league[league]["yd"].append(yd)
            by_league[league]["ya"].append(ya)

        # Fit global model
        if len(global_X) >= self._min_samples:
            self._global_reg.home.fit(global_X, global_yh)
            self._global_reg.draw.fit(global_X, global_yd)
            self._global_reg.away.fit(global_X, global_ya)
            self._global_reg.n_train = len(global_X)
            logger.info(
                "ClosingLineModel: global model fitted on %d samples.", len(global_X)
            )

        # Fit per-league models
        if self._league_split:
            for league, data in by_league.items():
                n = len(data["X"])
                if n < self._min_samples:
                    continue
                reg = _MarketRegressors(
                    home=_make_ridge(self._alpha),
                    draw=_make_ridge(self._alpha),
                    away=_make_ridge(self._alpha),
                    n_train=n,
                )
                reg.home.fit(data["X"], data["yh"])
                reg.draw.fit(data["X"], data["yd"])
                reg.away.fit(data["X"], data["ya"])
                self._regressors[league] = reg
                logger.debug(
                    "ClosingLineModel: league '%s' fitted on %d samples.", league, n
                )

        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_closing(
        self,
        opening_home: float,
        opening_draw: float,
        opening_away: float,
        league: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Predict expected closing odds for 1×2.

        Returns
        -------
        dict with keys: predicted_home, predicted_draw, predicted_away
        """
        feats = [_opening_features(opening_home, opening_draw, opening_away)]
        reg = self._pick_regressor(league)

        if not reg.home._fitted:
            # Fallback: return opening odds (no-model case)
            return {
                "predicted_home": opening_home,
                "predicted_draw": opening_draw,
                "predicted_away": opening_away,
            }

        ph = math.exp(reg.home.predict(feats)[0])
        pd_ = math.exp(reg.draw.predict(feats)[0])
        pa = math.exp(reg.away.predict(feats)[0])
        return {
            "predicted_home": round(max(1.01, ph), 4),
            "predicted_draw": round(max(1.01, pd_), 4),
            "predicted_away": round(max(1.01, pa), 4),
        }

    def clv(self, bet_odds: float, predicted_closing: float) -> float:
        """
        Closing Line Value in percentage points.

        CLV = 100 × [ln(bet_odds) − ln(closing_odds)]
        Positive → bet at better price than market closed at.
        """
        if predicted_closing <= 1.0 or bet_odds <= 1.0:
            return 0.0
        return 100.0 * (_log_odds(bet_odds) - _log_odds(predicted_closing))

    # ------------------------------------------------------------------
    # Batch scoring
    # ------------------------------------------------------------------

    def score_bets(self, bets: List[Dict]) -> List[Dict]:
        """
        Attach CLV fields to each bet dict.

        Expected keys per bet:
          fixture_id, market ("home"|"draw"|"away"), odds,
          opening_home_odds, opening_draw_odds, opening_away_odds
        Optional:
          league

        Returns a new list with CLV fields appended.
        """
        results = []
        for bet in bets:
            enriched = dict(bet)
            try:
                fid = str(bet.get("fixture_id", bet.get("match_id", "")))
                market = str(bet.get("market", "home")).lower()
                bet_odds = float(bet.get("odds", 0.0))
                oh = float(bet.get("opening_home_odds", bet_odds))
                od_ = float(bet.get("opening_draw_odds", bet_odds))
                oa = float(bet.get("opening_away_odds", bet_odds))
                league = bet.get("league")

                closing = self.predict_closing(oh, od_, oa, league)
                market_key = f"predicted_{market}"
                pred_c = closing.get(market_key, bet_odds)

                clv_pct = self.clv(bet_odds, pred_c)
                reg = self._pick_regressor(league)
                conf = _model_confidence(reg.n_train)

                result = CLVResult(
                    fixture_id=fid,
                    market=market,
                    bet_odds=bet_odds,
                    predicted_closing=pred_c,
                    clv_pct=clv_pct,
                    clv_grade=_clv_grade(clv_pct),
                    confidence=conf,
                )
                enriched.update(result.as_dict())
            except Exception as exc:
                logger.debug("ClosingLineModel.score_bets: skipping bet: %s", exc)
                enriched.setdefault("clv_pct", 0.0)
                enriched.setdefault("clv_grade", "NEUTRAL")
                enriched.setdefault("clv_confidence", 0.0)
            results.append(enriched)
        return results

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def summary(self) -> Dict:
        """Return training summary statistics."""
        league_counts = {lg: r.n_train for lg, r in self._regressors.items()}
        return {
            "global_n_train": self._global_reg.n_train,
            "global_fitted": self._global_reg.home._fitted,
            "n_leagues": len(self._regressors),
            "league_sample_counts": league_counts,
            "alpha": self._alpha,
            "min_samples": self._min_samples,
        }

    def evaluate(self, test_records: List[Dict]) -> Dict:
        """
        Evaluate model on held-out records.

        Returns MAE and RMSE for each market in log-odds space,
        plus mean absolute CLV prediction error.
        """
        errors = {"home": [], "draw": [], "away": []}
        clv_errors = []

        for r in test_records:
            try:
                oh = float(r["opening_home_odds"])
                od_ = float(r["opening_draw_odds"])
                oa = float(r["opening_away_odds"])
                ch = float(r["closing_home_odds"])
                cd = float(r["closing_draw_odds"])
                ca = float(r["closing_away_odds"])
            except (KeyError, TypeError, ValueError):
                continue

            league = r.get("league")
            pred = self.predict_closing(oh, od_, oa, league)

            errors["home"].append(
                abs(_log_odds(pred["predicted_home"]) - _log_odds(ch))
            )
            errors["draw"].append(
                abs(_log_odds(pred["predicted_draw"]) - _log_odds(cd))
            )
            errors["away"].append(
                abs(_log_odds(pred["predicted_away"]) - _log_odds(ca))
            )

            clv_h = abs(self.clv(oh, ch) - self.clv(pred["predicted_home"], ch))
            clv_errors.append(clv_h)

        def _stats(errs):
            if not errs:
                return {"mae": 0.0, "rmse": 0.0, "n": 0}
            n = len(errs)
            mae = sum(errs) / n
            rmse = math.sqrt(sum(e * e for e in errs) / n)
            return {"mae": round(mae, 6), "rmse": round(rmse, 6), "n": n}

        return {
            "home": _stats(errors["home"]),
            "draw": _stats(errors["draw"]),
            "away": _stats(errors["away"]),
            "clv_mae": round(sum(clv_errors) / len(clv_errors), 4)
            if clv_errors
            else 0.0,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _pick_regressor(self, league: Optional[str]) -> _MarketRegressors:
        """Return league-specific regressor if available, else global."""
        if league and self._league_split:
            reg = self._regressors.get(str(league))
            if reg and reg.home._fitted:
                return reg
        return self._global_reg

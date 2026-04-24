"""
Advanced AI feature generator for football match prediction.

Generates a rich feature set from raw match/team statistics using:
  1. Polynomial interaction features (degree-2 cross-products)
  2. Exponential weighted moving average (EWM) momentum
  3. Form streak encoding (win/draw/loss sequences → signed integer)
  4. Venue-split statistics (home vs. away performance separately)
  5. Volatility features (rolling std of outcome indicators)
  6. Head-to-head (H2H) dominance scores
  7. Mutual-information-based feature selection

Architecture
------------
FeatureGenerator
  .generate(team_records)  → pd.DataFrame (or list[dict])
  .select(df, target_col, k)  → list of top-k feature names by MI
  .importance(df, target_col)  → {feature: mi_score}

EWMTracker  — per-team exponential weighted averages
FormEncoder — converts last-N results to streak int + win/draw/loss rates
H2HAnalyser — head-to-head dominance between two teams

Mathematics
-----------
EWM:    s_t = α × x_t + (1 − α) × s_{t−1}    (α = 2 / (span + 1))
Streak: consecutive runs of W/D/L; +N for win streak, −N for loss streak
Gini impurity proxy for MI:
  MI(X; Y) ≈ H(Y) − H(Y|X_bin)   (discrete X bins, binary Y)
  H(Y) = −p log p − (1−p) log(1−p)
Polynomial: all pairs (x_i × x_j) for i ≤ j
Volatility: σ_k = std(x_{t−k:t}) over last k observations

Usage
-----
    from ai.feature_generator import FeatureGenerator

    gen = FeatureGenerator(ewm_spans=[5, 10, 20], poly_degree=2)
    df  = gen.generate(team_history_records)
    top = gen.select(df, target_col="home_win", k=20)
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional numpy / pandas
# ---------------------------------------------------------------------------

try:
    import importlib.util

    _PANDAS = (
        importlib.util.find_spec("pandas") is not None
        and importlib.util.find_spec("numpy") is not None
    )
except Exception:
    _PANDAS = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_SPANS = [3, 5, 10, 20]
_DEFAULT_WINDOWS = [3, 5]
_FORM_N = 5  # last N matches for form encoding
_MI_BINS = 10


# ---------------------------------------------------------------------------
# EWM Tracker
# ---------------------------------------------------------------------------


class EWMTracker:
    """
    Tracks exponential weighted moving averages for a stream of values.

    Supports multiple spans simultaneously.
    """

    def __init__(self, spans: List[int]) -> None:
        self._spans = spans
        self._alphas = {s: 2.0 / (s + 1) for s in spans}
        self._values: Dict[int, Optional[float]] = {s: None for s in spans}

    def update(self, x: float) -> Dict[str, float]:
        """Update all EWMs with new value x; return current EWM dict."""
        result = {}
        for s in self._spans:
            α = self._alphas[s]
            v = self._values[s]
            if v is None:
                self._values[s] = x
            else:
                self._values[s] = α * x + (1 - α) * v
            result[f"ewm_{s}"] = self._values[s]
        return result

    def reset(self) -> None:
        self._values = {s: None for s in self._spans}

    def current(self) -> Dict[str, Optional[float]]:
        return {f"ewm_{s}": self._values[s] for s in self._spans}


# ---------------------------------------------------------------------------
# Form Encoder
# ---------------------------------------------------------------------------


class FormEncoder:
    """
    Encodes recent form as:
    - streak: +N (win streak length), −N (loss streak), 0 (mixed/draw)
    - win_rate, draw_rate, loss_rate over last N matches
    - momentum: signed momentum = (wins − losses) / N
    """

    def __init__(self, n: int = _FORM_N) -> None:
        self._n = n
        self._history: deque = deque(maxlen=n)

    def record(self, result: str) -> None:
        """Record a result: 'W', 'D', or 'L'."""
        r = str(result).upper()
        if r not in ("W", "D", "L"):
            return
        self._history.append(r)

    def encode(self) -> Dict[str, float]:
        hist = list(self._history)
        n = len(hist)
        if n == 0:
            return {
                "form_streak": 0.0,
                "form_win_rate": 0.0,
                "form_draw_rate": 0.0,
                "form_loss_rate": 0.0,
                "form_momentum": 0.0,
                "form_n": 0.0,
            }

        wins = hist.count("W")
        draws = hist.count("D")
        losses = hist.count("L")

        # Streak: consecutive same-result run at end of history
        streak_val = hist[-1]
        streak_len = 0
        for r in reversed(hist):
            if r == streak_val:
                streak_len += 1
            else:
                break

        signed_streak = (
            streak_len
            if streak_val == "W"
            else (-streak_len if streak_val == "L" else 0)
        )

        return {
            "form_streak": float(signed_streak),
            "form_win_rate": wins / n,
            "form_draw_rate": draws / n,
            "form_loss_rate": losses / n,
            "form_momentum": (wins - losses) / n,
            "form_n": float(n),
        }

    def reset(self) -> None:
        self._history.clear()


# ---------------------------------------------------------------------------
# H2H Analyser
# ---------------------------------------------------------------------------


@dataclass
class H2HRecord:
    """Tracks head-to-head results between two teams."""

    wins_a: int = 0
    draws: int = 0
    wins_b: int = 0
    goals_a: int = 0
    goals_b: int = 0

    def record_match(self, result: str, goals_a: int = 0, goals_b: int = 0) -> None:
        self.goals_a += goals_a
        self.goals_b += goals_b
        r = result.upper()
        if r == "A":
            self.wins_a += 1
        elif r == "B":
            self.wins_b += 1
        else:
            self.draws += 1

    def dominance(self) -> Dict[str, float]:
        n = self.wins_a + self.draws + self.wins_b or 1
        return {
            "h2h_win_rate_a": self.wins_a / n,
            "h2h_draw_rate": self.draws / n,
            "h2h_win_rate_b": self.wins_b / n,
            "h2h_goal_diff": (self.goals_a - self.goals_b) / n,
            "h2h_n": float(n),
        }


class H2HAnalyser:
    """Maintains H2H records for all team pairs."""

    def __init__(self) -> None:
        self._records: Dict[Tuple[str, str], H2HRecord] = {}

    def _key(self, team_a: str, team_b: str) -> Tuple[str, str]:
        return (min(team_a, team_b), max(team_a, team_b))

    def record(
        self,
        team_a: str,
        team_b: str,
        result: str,
        goals_a: int = 0,
        goals_b: int = 0,
    ) -> None:
        k = self._key(team_a, team_b)
        if k not in self._records:
            self._records[k] = H2HRecord()
        # Normalise: always store as 'a' = lexicographically smaller team
        if team_a <= team_b:
            self._records[k].record_match(result, goals_a, goals_b)
        else:
            inv = {"A": "B", "B": "A", "D": "D"}.get(result.upper(), "D")
            self._records[k].record_match(inv, goals_b, goals_a)

    def features(self, home: str, away: str) -> Dict[str, float]:
        k = self._key(home, away)
        rec = self._records.get(k, H2HRecord())
        dom = rec.dominance()
        # Orient from home team's perspective
        if home > away:
            return {
                "h2h_home_win_rate": dom["h2h_win_rate_b"],
                "h2h_draw_rate": dom["h2h_draw_rate"],
                "h2h_away_win_rate": dom["h2h_win_rate_a"],
                "h2h_home_goal_diff": -dom["h2h_goal_diff"],
                "h2h_n": dom["h2h_n"],
            }
        return {
            "h2h_home_win_rate": dom["h2h_win_rate_a"],
            "h2h_draw_rate": dom["h2h_draw_rate"],
            "h2h_away_win_rate": dom["h2h_win_rate_b"],
            "h2h_home_goal_diff": dom["h2h_goal_diff"],
            "h2h_n": dom["h2h_n"],
        }


# ---------------------------------------------------------------------------
# Volatility tracker
# ---------------------------------------------------------------------------


class _RollingVol:
    """Rolling standard deviation (biased) over a fixed window."""

    def __init__(self, window: int) -> None:
        self._w = window
        self._buf: deque = deque(maxlen=window)

    def push(self, x: float) -> Optional[float]:
        self._buf.append(x)
        if len(self._buf) < 2:
            return None
        n = len(self._buf)
        mean = sum(self._buf) / n
        var = sum((v - mean) ** 2 for v in self._buf) / n
        return math.sqrt(var)

    def current(self) -> Optional[float]:
        if len(self._buf) < 2:
            return None
        n = len(self._buf)
        mean = sum(self._buf) / n
        var = sum((v - mean) ** 2 for v in self._buf) / n
        return math.sqrt(var)


# ---------------------------------------------------------------------------
# Polynomial features
# ---------------------------------------------------------------------------


def _polynomial_features(base: Dict[str, float], degree: int = 2) -> Dict[str, float]:
    """
    Generate degree-2 interaction terms for all numeric base features.

    Excludes self-squared terms to reduce multicollinearity.
    """
    if degree < 2:
        return {}
    keys = sorted(base.keys())
    interactions = {}
    for i, k1 in enumerate(keys):
        for k2 in keys[i + 1 :]:
            v1, v2 = base.get(k1, 0.0), base.get(k2, 0.0)
            if v1 is not None and v2 is not None:
                interactions[f"poly_{k1}_x_{k2}"] = float(v1) * float(v2)
    return interactions


# ---------------------------------------------------------------------------
# Mutual information (approximate, binned)
# ---------------------------------------------------------------------------


def _entropy(counts: List[int]) -> float:
    total = sum(counts)
    if total == 0:
        return 0.0
    h = 0.0
    for c in counts:
        if c > 0:
            p = c / total
            h -= p * math.log(p)
    return h


def _mutual_information(
    x_vals: List[float],
    y_vals: List[int],  # binary 0/1
    n_bins: int = _MI_BINS,
) -> float:
    """
    Approximate mutual information between a continuous feature and binary target.
    MI(X; Y) = H(Y) − H(Y|X)
    """
    if not x_vals or len(set(y_vals)) < 2:
        return 0.0

    # Marginal H(Y)
    n = len(y_vals)
    p1 = sum(y_vals) / n
    p0 = 1.0 - p1
    hy = _entropy([int(p0 * n), int(p1 * n)])

    if hy == 0:
        return 0.0

    # Bin x
    mn, mx = min(x_vals), max(x_vals)
    if mx == mn:
        return 0.0
    width = (mx - mn) / n_bins

    # H(Y|X)
    bins: Dict[int, List[int]] = defaultdict(list)
    for xv, yv in zip(x_vals, y_vals):
        b = min(int((xv - mn) / width), n_bins - 1)
        bins[b].append(yv)

    hyx = 0.0
    for b_vals in bins.values():
        nb = len(b_vals)
        pb = nb / n
        nb1 = sum(b_vals)
        nb0 = nb - nb1
        hyx += pb * _entropy([nb0, nb1])

    return max(0.0, hy - hyx)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class FeatureGenerator:
    """
    Advanced feature engineering pipeline for football prediction.

    Parameters
    ----------
    ewm_spans     : EWM half-life spans to compute (default [3,5,10,20])
    vol_windows   : rolling volatility windows (default [3,5])
    poly_degree   : polynomial interaction degree (default 2)
    form_n        : recent-form window size (default 5)
    include_poly  : whether to add interaction features (default True)
    """

    def __init__(
        self,
        ewm_spans: Optional[List[int]] = None,
        vol_windows: Optional[List[int]] = None,
        poly_degree: int = 2,
        form_n: int = _FORM_N,
        include_poly: bool = True,
    ) -> None:
        self._spans = ewm_spans or _DEFAULT_SPANS
        self._windows = vol_windows or _DEFAULT_WINDOWS
        self._poly_degree = poly_degree
        self._form_n = form_n
        self._include_poly = include_poly

        # Per-team state
        self._team_ewm: Dict[str, Dict[str, EWMTracker]] = {}
        self._team_form: Dict[str, FormEncoder] = {}
        self._team_vol: Dict[str, Dict[int, _RollingVol]] = {}
        self._team_venue_stats: Dict[str, Dict[str, Dict[str, float]]] = {}
        self._h2h = H2HAnalyser()

    # ------------------------------------------------------------------
    # Training: build team state from historical records
    # ------------------------------------------------------------------

    def fit(self, records: List[Dict]) -> "FeatureGenerator":
        """
        Fit EWM, form, and volatility trackers from chronological records.

        Each record must contain:
          - home_team, away_team
          - actual_outcome : "home_win" | "draw" | "away_win"
        Optional:
          - home_goals, away_goals
          - home_xg, away_xg (expected goals)
          - date or timestamp (for ordering)
        """
        sorted_recs = sorted(
            records,
            key=lambda r: str(r.get("date", r.get("timestamp", ""))),
        )

        for r in sorted_recs:
            home = str(r.get("home_team", ""))
            away = str(r.get("away_team", ""))
            outcome = str(r.get("actual_outcome", ""))
            if not home or not away or outcome not in ("home_win", "draw", "away_win"):
                continue

            hg = float(r.get("home_goals", 0))
            ag = float(r.get("away_goals", 0))
            hxg = float(r.get("home_xg", hg))
            axg = float(r.get("away_xg", ag))

            home_result = (
                "W"
                if outcome == "home_win"
                else ("L" if outcome == "away_win" else "D")
            )
            away_result = (
                "L"
                if outcome == "home_win"
                else ("W" if outcome == "away_win" else "D")
            )

            self._update_team(home, "home", home_result, hg, ag, hxg, axg)
            self._update_team(away, "away", away_result, ag, hg, axg, hxg)
            self._h2h.record(
                home,
                away,
                "A"
                if outcome == "home_win"
                else ("B" if outcome == "away_win" else "D"),
                int(hg),
                int(ag),
            )

        logger.info(
            "FeatureGenerator.fit: processed %d records for %d teams.",
            len(sorted_recs),
            len(self._team_ewm),
        )
        return self

    def _update_team(
        self,
        team: str,
        venue: str,
        result: str,
        gf: float,
        ga: float,
        xgf: float,
        xga: float,
    ) -> None:
        # EWM trackers
        if team not in self._team_ewm:
            self._team_ewm[team] = {
                "goals_for": EWMTracker(self._spans),
                "goals_against": EWMTracker(self._spans),
                "xg_for": EWMTracker(self._spans),
                "xg_against": EWMTracker(self._spans),
                "goal_diff": EWMTracker(self._spans),
            }
        ewms = self._team_ewm[team]
        ewms["goals_for"].update(gf)
        ewms["goals_against"].update(ga)
        ewms["xg_for"].update(xgf)
        ewms["xg_against"].update(xga)
        ewms["goal_diff"].update(gf - ga)

        # Form
        if team not in self._team_form:
            self._team_form[team] = FormEncoder(self._form_n)
        self._team_form[team].record(result)

        # Volatility
        if team not in self._team_vol:
            self._team_vol[team] = {w: _RollingVol(w) for w in self._windows}
        win_indicator = 1.0 if result == "W" else 0.0
        for vol in self._team_vol[team].values():
            vol.push(win_indicator)

        # Venue stats
        if team not in self._team_venue_stats:
            self._team_venue_stats[team] = {
                "home": defaultdict(float),
                "away": defaultdict(float),
            }
        vs = self._team_venue_stats[team][venue]
        vs["n"] = vs.get("n", 0) + 1
        vs["gf_sum"] = vs.get("gf_sum", 0.0) + gf
        vs["ga_sum"] = vs.get("ga_sum", 0.0) + ga
        vs["wins"] = vs.get("wins", 0) + (1 if result == "W" else 0)
        vs["draws"] = vs.get("draws", 0) + (1 if result == "D" else 0)

    # ------------------------------------------------------------------
    # Feature generation for a match
    # ------------------------------------------------------------------

    def generate_match_features(self, home: str, away: str) -> Dict[str, float]:
        """
        Generate all features for a single upcoming match.

        Returns a flat dict of feature_name → float.
        """
        feats: Dict[str, float] = {}

        feats.update(self._ewm_features(home, "home"))
        feats.update(self._ewm_features(away, "away"))
        feats.update(self._form_features(home, "home"))
        feats.update(self._form_features(away, "away"))
        feats.update(self._vol_features(home, "home"))
        feats.update(self._vol_features(away, "away"))
        feats.update(self._venue_features(home, "home"))
        feats.update(self._venue_features(away, "away"))
        feats.update(self._h2h.features(home, away))
        feats.update(self._differential_features(feats))

        if self._include_poly:
            # Limit poly to the ~20 most informative base features
            base_for_poly = {
                k: v for k, v in feats.items() if not k.startswith("poly_")
            }
            feats.update(_polynomial_features(base_for_poly, self._poly_degree))

        return {k: (float(v) if v is not None else 0.0) for k, v in feats.items()}

    def generate(self, records: List[Dict]) -> List[Dict]:
        """
        Generate features for a list of match records.

        Each record must contain home_team and away_team.
        """
        out = []
        for r in records:
            home = str(r.get("home_team", ""))
            away = str(r.get("away_team", ""))
            feats = self.generate_match_features(home, away)
            merged = {**r, **feats}
            out.append(merged)
        return out

    # ------------------------------------------------------------------
    # Feature selection via mutual information
    # ------------------------------------------------------------------

    def select(
        self,
        records: List[Dict],
        target_col: str,
        k: int = 20,
    ) -> List[str]:
        """
        Return top-k feature names ranked by mutual information with target.

        target_col must be binary (0/1) or convertible.
        """
        scores = self.importance(records, target_col)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [name for name, _ in ranked[:k]]

    def importance(
        self,
        records: List[Dict],
        target_col: str,
    ) -> Dict[str, float]:
        """Compute mutual information of each feature with the binary target."""
        if not records:
            return {}

        target_vals = [int(bool(r.get(target_col, 0))) for r in records]
        if len(set(target_vals)) < 2:
            logger.warning(
                "FeatureGenerator.importance: target '%s' has no variance.", target_col
            )
            return {}

        all_keys = set()
        for r in records:
            all_keys.update(r.keys())
        all_keys.discard(target_col)

        mi_scores: Dict[str, float] = {}
        for key in all_keys:
            try:
                vals = [float(r.get(key, 0.0) or 0.0) for r in records]
                mi = _mutual_information(vals, target_vals, _MI_BINS)
                mi_scores[key] = round(mi, 6)
            except Exception:
                pass

        return dict(sorted(mi_scores.items(), key=lambda x: x[1], reverse=True))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ewm_features(self, team: str, prefix: str) -> Dict[str, float]:
        if team not in self._team_ewm:
            return {}
        feats = {}
        for metric, tracker in self._team_ewm[team].items():
            for span_key, val in tracker.current().items():
                feats[f"{prefix}_{metric}_{span_key}"] = val if val is not None else 0.0
        return feats

    def _form_features(self, team: str, prefix: str) -> Dict[str, float]:
        if team not in self._team_form:
            return {
                f"{prefix}_form_streak": 0.0,
                f"{prefix}_form_win_rate": 0.0,
                f"{prefix}_form_draw_rate": 0.0,
                f"{prefix}_form_loss_rate": 0.0,
                f"{prefix}_form_momentum": 0.0,
            }
        enc = self._team_form[team].encode()
        return {f"{prefix}_{k}": v for k, v in enc.items()}

    def _vol_features(self, team: str, prefix: str) -> Dict[str, float]:
        if team not in self._team_vol:
            return {}
        feats = {}
        for window, tracker in self._team_vol[team].items():
            val = tracker.current()
            feats[f"{prefix}_vol_w{window}"] = val if val is not None else 0.0
        return feats

    def _venue_features(self, team: str, venue: str) -> Dict[str, float]:
        if team not in self._team_venue_stats:
            return {}
        vs = self._team_venue_stats[team].get(venue, {})
        n = vs.get("n", 0) or 1
        prefix = f"{venue}_venue"
        return {
            f"{prefix}_win_rate": vs.get("wins", 0) / n,
            f"{prefix}_draw_rate": vs.get("draws", 0) / n,
            f"{prefix}_avg_gf": vs.get("gf_sum", 0.0) / n,
            f"{prefix}_avg_ga": vs.get("ga_sum", 0.0) / n,
            f"{prefix}_avg_gd": (vs.get("gf_sum", 0.0) - vs.get("ga_sum", 0.0)) / n,
            f"{venue}_venue_n": float(n),
        }

    @staticmethod
    def _differential_features(feats: Dict[str, float]) -> Dict[str, float]:
        """
        Compute home − away differentials for comparable metrics.
        """
        diffs = {}
        home_keys = {k[5:]: v for k, v in feats.items() if k.startswith("home_")}
        away_keys = {k[5:]: v for k, v in feats.items() if k.startswith("away_")}
        for k in home_keys:
            if k in away_keys:
                diffs[f"diff_{k}"] = home_keys[k] - away_keys[k]
        return diffs

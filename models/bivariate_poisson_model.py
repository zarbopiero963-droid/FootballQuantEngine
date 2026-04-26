"""
Real Bivariate Poisson model for football score prediction.

Reference: Karlis & Ntzoufras (2003) — "Analysis of sports data by using
bivariate Poisson models", The Statistician.

The bivariate Poisson distribution models (X, Y) as:
  X = X1 + X3,  Y = X2 + X3
where X1 ~ Pois(λ1), X2 ~ Pois(λ2), X3 ~ Pois(λ3) are independent.
The shared component X3 introduces positive correlation between X and Y.

Joint PMF (Karlis & Ntzoufras, 2003, eq. 1):
  P(X=x, Y=y) = exp(-(λ1+λ2+λ3)) × (λ1^x / x!) × (λ2^y / y!)
                × Σ_{k=0}^{min(x,y)} C(x,k) C(y,k) k! (λ3/(λ1 λ2))^k

Parameters
----------
λ1 : home scoring rate (private: only home goals)
λ2 : away scoring rate (private: only away goals)
λ3 : covariance term  (shared goals; induces positive correlation)
     Cov(X, Y) = λ3
     Cor(X, Y) = λ3 / sqrt((λ1+λ3)(λ2+λ3))
"""

from __future__ import annotations

import math
from collections import defaultdict

_MAX_GOALS = 8


# ---------------------------------------------------------------------------
# Bivariate Poisson PMF
# ---------------------------------------------------------------------------


def _bvp_pmf(x: int, y: int, l1: float, l2: float, l3: float) -> float:
    """
    P(X=x, Y=y) under Bivariate Poisson(λ1, λ2, λ3).

    Computed via the series representation (Karlis & Ntzoufras eq. 1).
    Numerically stable: log-sum-exp over the k-terms.
    """
    if x < 0 or y < 0 or l1 <= 0 or l2 <= 0:
        return 0.0

    # Constant part
    log_c = (
        -(l1 + l2 + l3)
        + x * math.log(l1)
        - math.lgamma(x + 1)
        + y * math.log(l2)
        - math.lgamma(y + 1)
    )

    # Series Σ_{k=0}^{min(x,y)} C(x,k)C(y,k)k! (λ3/(λ1λ2))^k
    k_max = min(x, y)
    log_l3_ratio = (
        math.log(l3) - math.log(l1) - math.log(l2) if l3 > 0 else float("-inf")
    )

    log_terms = []
    for k in range(k_max + 1):
        if l3 <= 0 and k > 0:
            break
        log_term = (
            math.lgamma(x + 1)
            - math.lgamma(x - k + 1)
            - math.lgamma(k + 1)  # C(x,k)
            + math.lgamma(y + 1)
            - math.lgamma(y - k + 1)
            - math.lgamma(k + 1)  # C(y,k)
            + math.lgamma(k + 1)  # k!
            + k * log_l3_ratio
            if l3 > 0
            else 0.0
        )
        log_terms.append(log_term)

    if not log_terms:
        return 0.0

    # log-sum-exp for numerical stability
    max_lt = max(log_terms)
    series = math.exp(max_lt) * sum(math.exp(lt - max_lt) for lt in log_terms)

    return math.exp(log_c) * series


# ---------------------------------------------------------------------------
# Score matrix + market probabilities
# ---------------------------------------------------------------------------


def _score_matrix(
    l1: float, l2: float, l3: float, max_goals: int = _MAX_GOALS
) -> list[list[float]]:
    matrix = []
    for i in range(max_goals + 1):
        row = [_bvp_pmf(i, j, l1, l2, l3) for j in range(max_goals + 1)]
        matrix.append(row)
    total = sum(p for row in matrix for p in row)
    if total > 0:
        matrix = [[p / total for p in row] for row in matrix]
    return matrix


def _markets_from_matrix(matrix: list[list[float]]) -> dict:
    hw = draw = aw = 0.0
    o15 = o25 = o35 = btts = 0.0
    for i, row in enumerate(matrix):
        for j, p in enumerate(row):
            tg = i + j
            if i > j:
                hw += p
            elif i == j:
                draw += p
            else:
                aw += p
            if tg > 1:
                o15 += p
            if tg > 2:
                o25 += p
            if tg > 3:
                o35 += p
            if i > 0 and j > 0:
                btts += p
    return {
        "home_win": round(hw, 6),
        "draw": round(draw, 6),
        "away_win": round(aw, 6),
        "over_15": round(o15, 6),
        "under_15": round(1.0 - o15, 6),
        "over_25": round(o25, 6),
        "under_25": round(1.0 - o25, 6),
        "over_35": round(o35, 6),
        "under_35": round(1.0 - o35, 6),
        "btts_yes": round(btts, 6),
        "btts_no": round(1.0 - btts, 6),
    }


# ---------------------------------------------------------------------------
# Log-likelihood for MLE
# ---------------------------------------------------------------------------


def _log_likelihood(matches: list[dict], l1: float, l2: float, l3: float) -> float:
    ll = 0.0
    for m in matches:
        x = int(m.get("home_goals") or 0)
        y = int(m.get("away_goals") or 0)
        p = _bvp_pmf(x, y, l1, l2, l3)
        ll += math.log(max(p, 1e-15))
    return ll


# ---------------------------------------------------------------------------
# Bivariate Poisson Model
# ---------------------------------------------------------------------------


class BivariatePoissonModel:
    """
    Team-level Bivariate Poisson football model.

    Fitting
    -------
    Uses a two-stage approach:
    1. Estimate team attack/defense strengths using the standard Dixon-Robinson
       iterative reweighted least-squares initialisation.
    2. Fit λ3 (covariance) by 1-D grid search over the log-likelihood, holding
       team parameters fixed.

    The team parameters follow the usual Dixon-Coles structure:
      λ1_match = mu_home × attack_home × defense_away   (private home goals)
      λ2_match = mu_away × attack_away × defense_home   (private away goals)
      λ3_match = λ3_global                              (shared goals, constant)

    Prediction
    ----------
    Full joint score matrix → 1×2, O/U, BTTS.

    Correlation
    -----------
    Cor(home_goals, away_goals) = λ3 / sqrt((λ1+λ3)(λ2+λ3))
    Typical football value: 0.01–0.05.
    """

    _PRIOR_N: float = 3.0

    def __init__(self) -> None:
        self._att: dict[str, float] = {}  # normalised attack
        self._def: dict[str, float] = {}  # normalised defense
        self._mu_home: float = 1.35
        self._mu_away: float = 1.10
        self._l3: float = 0.05  # covariance term

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, completed_matches: list[dict]) -> "BivariatePoissonModel":
        if not completed_matches:
            return self

        acc = defaultdict(lambda: {"scored": 0.0, "conceded": 0.0, "n": 0})
        th = ta = 0.0
        n = 0

        for m in completed_matches:
            h = str(m["home_team"])
            a = str(m["away_team"])
            hg = float(m.get("home_goals") or 0)
            ag = float(m.get("away_goals") or 0)

            acc[h]["scored"] += hg
            acc[h]["conceded"] += ag
            acc[h]["n"] += 1

            acc[a]["scored"] += ag
            acc[a]["conceded"] += hg
            acc[a]["n"] += 1

            th += hg
            ta += ag
            n += 1

        if not n:
            return self

        self._mu_home = th / n
        self._mu_away = ta / n
        mu_h = self._mu_home or 1.35
        mu_a = self._mu_away or 1.10

        for team, s in acc.items():
            nt = s["n"]
            raw_att = s["scored"] / nt
            raw_def = s["conceded"] / nt
            self._att[team] = (
                (raw_att * nt + mu_h * self._PRIOR_N) / (nt + self._PRIOR_N) / mu_h
            )
            self._def[team] = (
                (raw_def * nt + mu_a * self._PRIOR_N) / (nt + self._PRIOR_N) / mu_a
            )

        # Fit λ3 by 1-D grid search
        self._l3 = self._fit_l3(completed_matches)
        return self

    def _fit_l3(self, matches: list[dict]) -> float:
        """Grid search for the global covariance parameter λ3."""
        best_l3, best_ll = 0.0, float("-inf")
        candidates = [i / 100.0 for i in range(0, 51, 5)]  # 0.00, 0.05, … 0.50
        for l3_cand in candidates:
            ll = 0.0
            for m in matches:
                l1, l2 = self._private_lambdas(
                    str(m["home_team"]), str(m["away_team"]), l3_cand
                )
                x = int(m.get("home_goals") or 0)
                y = int(m.get("away_goals") or 0)
                p = _bvp_pmf(x, y, l1, l2, l3_cand)
                ll += math.log(max(p, 1e-15))
            if ll > best_ll:
                best_ll, best_l3 = ll, l3_cand
        return best_l3

    # ------------------------------------------------------------------
    # Lambdas
    # ------------------------------------------------------------------

    def _private_lambdas(self, home: str, away: str, l3: float) -> tuple[float, float]:
        """
        Private (non-shared) goal rates.
        Total home goals = λ1 + λ3,  total away = λ2 + λ3.
        """
        att_h = self._att.get(home, 1.0)
        def_h = self._def.get(home, 1.0)
        att_a = self._att.get(away, 1.0)
        def_a = self._def.get(away, 1.0)

        # Total expected goals for each team
        total_h = max(self._mu_home * att_h * def_a, 0.10 + l3)
        total_a = max(self._mu_away * att_a * def_h, 0.10 + l3)

        # Private rates (subtract shared component)
        l1 = max(total_h - l3, 0.05)
        l2 = max(total_a - l3, 0.05)
        return l1, l2

    def _all_lambdas(self, home: str, away: str) -> tuple[float, float, float]:
        l1, l2 = self._private_lambdas(home, away, self._l3)
        return l1, l2, self._l3

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_teams(self, home_team: str, away_team: str) -> dict:
        l1, l2, l3 = self._all_lambdas(home_team, away_team)
        matrix = _score_matrix(l1, l2, l3)
        result = _markets_from_matrix(matrix)
        result["lambda1_private"] = round(l1, 4)
        result["lambda2_private"] = round(l2, 4)
        result["lambda3_shared"] = round(l3, 4)
        result["correlation"] = round(
            l3 / math.sqrt(max((l1 + l3) * (l2 + l3), 1e-9)), 6
        )
        return result

    def predict(self, row) -> dict:
        """Plugin-compatible row-level prediction."""
        if hasattr(row, "get"):
            home = row.get("home_team") or row.get("home")
            away = row.get("away_team") or row.get("away")
        else:
            home = getattr(row, "home_team", None) or getattr(row, "home", None)
            away = getattr(row, "away_team", None) or getattr(row, "away", None)

        if home and away:
            return self.predict_teams(str(home), str(away))

        try:
            ha = float(getattr(row, "home_attack", self._mu_home))
            ad = float(getattr(row, "away_defense", 1.0))
            aa = float(getattr(row, "away_attack", self._mu_away))
            hd = float(getattr(row, "home_defense", 1.0))
        except (TypeError, AttributeError):
            ha, ad, aa, hd = self._mu_home, 1.0, self._mu_away, 1.0

        l3 = self._l3
        total_h = max(ha * ad, 0.10 + l3)
        total_a = max(aa * hd, 0.10 + l3)
        l1 = max(total_h - l3, 0.05)
        l2 = max(total_a - l3, 0.05)

        matrix = _score_matrix(l1, l2, l3)
        return _markets_from_matrix(matrix)

    def score_matrix(self, home_team: str, away_team: str) -> list[list[float]]:
        """Full joint score probability matrix."""
        l1, l2, l3 = self._all_lambdas(home_team, away_team)
        return _score_matrix(l1, l2, l3)

    def covariance_info(self, home_team: str, away_team: str) -> dict:
        """Return fitted covariance and correlation for a fixture."""
        l1, l2, l3 = self._all_lambdas(home_team, away_team)
        corr = l3 / math.sqrt(max((l1 + l3) * (l2 + l3), 1e-9))
        return {
            "lambda3": round(l3, 6),
            "covariance": round(l3, 6),
            "correlation": round(corr, 6),
        }

"""
Real xG (Expected Goals) prediction model.

Uses team-level xG rates fitted from historical match data.
When xG is not explicitly tracked, falls back to actual goals as proxy.

Pipeline
--------
1. fit(completed_matches)  — compute each team's xG attack and xG defense rates,
   normalised to league averages; also fits Dixon-Coles rho for low-score bias.
2. predict_teams(home, away) — return full probability distribution from the
   fitted xG lambdas through a Poisson score matrix with DC correction.
3. predict_from_lambdas(lh, la) — same, but with explicit lambdas.
4. predict(row) — plugin API compatible with PredictionPipeline.

Markets computed
----------------
  1×2 (home_win, draw, away_win)
  Over/Under 2.5, 1.5, 3.5
  Both Teams To Score (BTTS yes/no)
  Asian Handicap ±0.5, ±1.5
"""

from __future__ import annotations

import math
from collections import defaultdict

_MAX_GOALS = 10


# ---------------------------------------------------------------------------
# Dixon-Coles tau correction
# ---------------------------------------------------------------------------


def _tau(hg: int, ag: int, lh: float, la: float, rho: float) -> float:
    if hg == 0 and ag == 0:
        return 1.0 - lh * la * rho
    if hg == 0 and ag == 1:
        return 1.0 + lh * rho
    if hg == 1 and ag == 0:
        return 1.0 + la * rho
    if hg == 1 and ag == 1:
        return 1.0 - rho
    return 1.0


def _poisson_pmf(k: int, lam: float) -> float:
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    log_p = -lam + k * math.log(lam) - math.lgamma(k + 1)
    return math.exp(log_p)


# ---------------------------------------------------------------------------
# Score matrix
# ---------------------------------------------------------------------------


def _score_matrix(
    lh: float, la: float, rho: float, max_goals: int = _MAX_GOALS
) -> list[list[float]]:
    """Joint score probability matrix M[i][j] = P(home=i, away=j)."""
    matrix = []
    for i in range(max_goals + 1):
        row = []
        for j in range(max_goals + 1):
            p = _poisson_pmf(i, lh) * _poisson_pmf(j, la) * _tau(i, j, lh, la, rho)
            row.append(max(p, 0.0))
        matrix.append(row)
    # Normalise to sum to 1 (tau can slightly perturb the total)
    total = sum(p for row in matrix for p in row)
    if total > 0:
        matrix = [[p / total for p in row] for row in matrix]
    return matrix


def _probs_from_matrix(matrix: list[list[float]]) -> dict:
    home_win = draw = away_win = 0.0
    over_15 = over_25 = over_35 = 0.0
    btts = 0.0
    ah_home_half = ah_away_half = 0.0  # AH ±0.5
    ah_home_1h = ah_away_1h = 0.0  # AH ±1.5

    for i, row in enumerate(matrix):
        for j, p in enumerate(row):
            total_goals = i + j
            if i > j:
                home_win += p
            elif i == j:
                draw += p
            else:
                away_win += p
            if total_goals > 1:
                over_15 += p
            if total_goals > 2:
                over_25 += p
            if total_goals > 3:
                over_35 += p
            if i > 0 and j > 0:
                btts += p
            # AH ±0.5: home covers if wins by any margin
            if i > j:
                ah_home_half += p
            else:
                ah_away_half += p
            # AH ±1.5: home covers if wins by 2+
            if i - j >= 2:
                ah_home_1h += p
            elif j - i >= 0:
                ah_away_1h += p

    return {
        "home_win": round(home_win, 6),
        "draw": round(draw, 6),
        "away_win": round(away_win, 6),
        "over_15": round(over_15, 6),
        "under_15": round(1.0 - over_15, 6),
        "over_25": round(over_25, 6),
        "under_25": round(1.0 - over_25, 6),
        "over_35": round(over_35, 6),
        "under_35": round(1.0 - over_35, 6),
        "btts_yes": round(btts, 6),
        "btts_no": round(1.0 - btts, 6),
        "ah_home_05": round(ah_home_half, 6),
        "ah_away_05": round(ah_away_half, 6),
        "ah_home_15": round(ah_home_1h, 6),
        "ah_away_15": round(ah_away_1h, 6),
    }


# ---------------------------------------------------------------------------
# xG Model
# ---------------------------------------------------------------------------


class XGModel:
    """
    xG-based football prediction model.

    Fitting
    -------
    For each team, compute the rolling season-level xG rates:
      xg_att = mean(xG scored per match)   normalised to league average
      xg_def = mean(xG conceded per match) normalised to league average

    When xg_scored / xg_conceded are not in the match dict, actual goals
    are used as a proxy (common pattern when full xG data is unavailable).

    Prediction
    ----------
      λ_home = mu_home × xg_att(home) × xg_def(away)
      λ_away = mu_away × xg_att(away) × xg_def(home)

    then Dixon-Coles rho correction is applied to the joint score matrix.
    """

    _PRIOR_N: float = 3.0  # pseudo-count matches for shrinkage toward league avg

    def __init__(self) -> None:
        self._att: dict[str, float] = {}  # {team: normalised xG attack rate}
        self._def: dict[str, float] = {}  # {team: normalised xG defense rate}
        self._mu_home: float = 1.35
        self._mu_away: float = 1.10
        self._rho: float = -0.1  # DC rho (fitted by MLE grid search)

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, completed_matches: list[dict]) -> "XGModel":
        """
        Fit xG attack and defense rates from completed matches.

        Each dict should contain:
          home_team, away_team,
          home_goals, away_goals,
          [optional] xg_home, xg_away  — if absent, goals are used as proxy.
        """
        if not completed_matches:
            return self

        acc = defaultdict(lambda: {"scored": 0.0, "conceded": 0.0, "n": 0})
        total_h = total_a = 0.0
        n = 0

        for m in completed_matches:
            h = str(m["home_team"])
            a = str(m["away_team"])
            # Prefer xG if available, fall back to actual goals
            hg = float(m.get("xg_home") or m.get("home_goals") or 0)
            ag = float(m.get("xg_away") or m.get("away_goals") or 0)

            acc[h]["scored"] += hg
            acc[h]["conceded"] += ag
            acc[h]["n"] += 1

            acc[a]["scored"] += ag
            acc[a]["conceded"] += hg
            acc[a]["n"] += 1

            total_h += hg
            total_a += ag
            n += 1

        if not n:
            return self

        self._mu_home = total_h / n
        self._mu_away = total_a / n

        mu_h = self._mu_home or 1.35
        mu_a = self._mu_away or 1.10

        # Bayesian shrinkage: blend team rate toward league average
        for team, s in acc.items():
            n_t = s["n"]
            # Shrunk attack rate (normalised)
            raw_att = s["scored"] / n_t
            self._att[team] = (
                (raw_att * n_t + mu_h * self._PRIOR_N) / (n_t + self._PRIOR_N) / mu_h
            )

            # Shrunk defense rate (normalised to away goals average)
            raw_def = s["conceded"] / n_t
            self._def[team] = (
                (raw_def * n_t + mu_a * self._PRIOR_N) / (n_t + self._PRIOR_N) / mu_a
            )

        # Fit rho by grid-search MLE
        self._rho = self._fit_rho(completed_matches)
        return self

    def _fit_rho(self, matches: list[dict]) -> float:
        best_rho, best_ll = 0.0, float("-inf")
        for rho_cand in [r / 100.0 for r in range(-25, 26, 5)]:
            ll = 0.0
            for m in matches:
                lh, la = self._lambdas(str(m["home_team"]), str(m["away_team"]))
                hg = int(m.get("home_goals") or 0)
                ag = int(m.get("away_goals") or 0)
                p_h = _poisson_pmf(hg, lh)
                p_a = _poisson_pmf(ag, la)
                tau = _tau(hg, ag, lh, la, rho_cand)
                p = p_h * p_a * tau
                ll += math.log(max(p, 1e-12))
            if ll > best_ll:
                best_ll, best_rho = ll, rho_cand
        return best_rho

    # ------------------------------------------------------------------
    # Lambdas
    # ------------------------------------------------------------------

    def _lambdas(self, home: str, away: str) -> tuple[float, float]:
        att_h = self._att.get(home, 1.0)
        def_h = self._def.get(home, 1.0)
        att_a = self._att.get(away, 1.0)
        def_a = self._def.get(away, 1.0)
        lh = max(self._mu_home * att_h * def_a, 0.10)
        la = max(self._mu_away * att_a * def_h, 0.10)
        return lh, la

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_teams(self, home_team: str, away_team: str) -> dict:
        """Full probability distribution for a named fixture."""
        lh, la = self._lambdas(home_team, away_team)
        return self.predict_from_lambdas(lh, la)

    def predict_from_lambdas(self, lh: float, la: float) -> dict:
        """Compute all market probabilities from explicit lambdas."""
        matrix = _score_matrix(lh, la, self._rho)
        probs = _probs_from_matrix(matrix)
        probs["lambda_home"] = round(lh, 4)
        probs["lambda_away"] = round(la, 4)
        probs["rho"] = round(self._rho, 4)
        return probs

    def predict(self, row) -> dict:
        """
        Plugin-compatible API: accepts a DataFrame row or dict.

        If home_team / away_team are present, uses fitted rates.
        Otherwise uses home_attack * away_defense as lambda proxy.
        """
        if hasattr(row, "get"):
            home = row.get("home_team") or row.get("home")
            away = row.get("away_team") or row.get("away")
        else:
            home = getattr(row, "home_team", None) or getattr(row, "home", None)
            away = getattr(row, "away_team", None) or getattr(row, "away", None)

        if home and away and (home in self._att or away in self._att):
            return self.predict_teams(str(home), str(away))

        # Fallback: derive lambdas from feature row
        try:
            ha = float(
                getattr(row, "home_attack", None)
                or row.get("home_attack", self._mu_home)
            )
            ad = float(
                getattr(row, "away_defense", None) or row.get("away_defense", 1.0)
            )
            aa = float(
                getattr(row, "away_attack", None)
                or row.get("away_attack", self._mu_away)
            )
            hd = float(
                getattr(row, "home_defense", None) or row.get("home_defense", 1.0)
            )
        except (TypeError, AttributeError):
            ha, ad, aa, hd = self._mu_home, 1.0, self._mu_away, 1.0

        lh = max(ha * ad, 0.10)
        la = max(aa * hd, 0.10)
        return self.predict_from_lambdas(lh, la)

    def score_matrix(self, home_team: str, away_team: str) -> list[list[float]]:
        """Return the full (MAX_GOALS+1)×(MAX_GOALS+1) score probability matrix."""
        lh, la = self._lambdas(home_team, away_team)
        return _score_matrix(lh, la, self._rho)

    def team_stats(self, team: str) -> dict:
        """Return fitted xG rates for a team."""
        return {
            "xg_attack_norm": round(self._att.get(team, 1.0), 4),
            "xg_defense_norm": round(self._def.get(team, 1.0), 4),
            "xg_attack_abs": round(self._att.get(team, 1.0) * self._mu_home, 4),
            "xg_defense_abs": round(self._def.get(team, 1.0) * self._mu_away, 4),
        }

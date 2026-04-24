"""
Gamma-Poisson conjugate Bayesian model for football goal scoring.

Each team's attacking rate λ_att ~ Gamma(α_att, β_att) and defensive
concession rate λ_def ~ Gamma(α_def, β_def). After observing N matches,
posteriors are updated analytically (conjugate pair). The posterior
predictive distribution for future goals is Negative Binomial, enabling
closed-form 1×2 probability computation without Monte Carlo sampling.

References:
  Dixon & Coles (1997) — baseline Poisson model
  Rue & Salvesen (2000) — Bayesian dynamic updating for football
  Karlis & Ntzoufras (2003) — Bivariate Poisson / Bayesian approach
"""

from __future__ import annotations

import math
from collections import defaultdict

# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------


def _neg_binom_pmf(k: int, r: float, p: float) -> float:
    """
    Negative Binomial PMF: P(X = k | r, p).

    Parameterisation: r = shape, p = success probability per trial.
    Mean = r·p / (1-p).
    """
    if k < 0 or r <= 0.0 or not (0.0 < p < 1.0):
        return 0.0
    log_pmf = (
        math.lgamma(k + r)
        - math.lgamma(k + 1)
        - math.lgamma(r)
        + r * math.log(1.0 - p)
        + k * math.log(p)
    )
    return math.exp(log_pmf)


def _normal_quantile(p: float) -> float:
    """Inverse normal CDF via Peter Acklam's rational approximation."""
    if p <= 0.0:
        return float("-inf")
    if p >= 1.0:
        return float("inf")
    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    ]
    p_lo, p_hi = 0.02425, 1.0 - 0.02425
    if p_lo <= p <= p_hi:
        q = p - 0.5
        r = q * q
        num = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
        den = ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1
        return num / den
    elif p < p_lo:
        q = math.sqrt(-2 * math.log(p))
        num = ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
        den = (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        return num / den
    else:
        q = math.sqrt(-2 * math.log(1.0 - p))
        num = ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
        den = (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        return -(num / den)


# ---------------------------------------------------------------------------
# Bayesian Model
# ---------------------------------------------------------------------------


class BayesianModel:
    """
    Conjugate Gamma-Poisson Bayesian football model.

    Prior
    -----
    For every team t:
      attack rate:  λ_att(t) ~ Gamma(α0, β0)   prior mean = α0/β0 = 1.0
      defense rate: λ_def(t) ~ Gamma(α0, β0)   (average team)

    Update rule  (exact conjugate)
    ------------------------------
    After observing team t score g goals in n matches:
      α_att ← α0 + Σ goals_scored
      β_att ← β0 + n

    Posterior predictive
    --------------------
    Marginalising λ_att ~ Gamma(α_att, β_att) out of Poisson(λ_att · μ_eff):

      goals ~ NegBin(r = α_att,  p = μ_eff / (α_att + μ_eff))

    where μ_eff = μ_venue × E[att(A)] × E[def(B)]  (expected goals for matchup).

    1×2 probabilities are computed from the full joint score distribution
    (NegBin_home × NegBin_away) summed over all score grids up to MAX_GOALS.
    """

    _PRIOR_ALPHA: float = 2.0
    _PRIOR_BETA: float = 2.0
    _MAX_GOALS: int = 10

    def __init__(self) -> None:
        self._att: dict[str, tuple[float, float]] = {}
        self._def: dict[str, tuple[float, float]] = {}
        self._league_mu_home: float = 1.35
        self._league_mu_away: float = 1.10

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, completed_matches: list[dict]) -> "BayesianModel":
        """
        Update Gamma posteriors from completed match list (batch mode).

        Each dict: {home_team, away_team, home_goals, away_goals}.
        """
        if not completed_matches:
            return self

        acc: dict[str, dict] = defaultdict(
            lambda: {"scored": 0.0, "conceded": 0.0, "n": 0}
        )
        total_home = total_away = 0.0
        n_total = 0

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

            total_home += hg
            total_away += ag
            n_total += 1

        if n_total:
            self._league_mu_home = total_home / n_total
            self._league_mu_away = total_away / n_total

        for team, s in acc.items():
            self._att[team] = (
                self._PRIOR_ALPHA + s["scored"],
                self._PRIOR_BETA + s["n"],
            )
            self._def[team] = (
                self._PRIOR_ALPHA + s["conceded"],
                self._PRIOR_BETA + s["n"],
            )

        return self

    def update_one(
        self,
        home_team: str,
        away_team: str,
        home_goals: int,
        away_goals: int,
    ) -> None:
        """Incremental single-match posterior update (online inference)."""
        for team, scored, conceded in (
            (home_team, home_goals, away_goals),
            (away_team, away_goals, home_goals),
        ):
            a_att, b_att = self._att.get(team, (self._PRIOR_ALPHA, self._PRIOR_BETA))
            a_def, b_def = self._def.get(team, (self._PRIOR_ALPHA, self._PRIOR_BETA))
            self._att[team] = (a_att + float(scored), b_att + 1.0)
            self._def[team] = (a_def + float(conceded), b_def + 1.0)

    # ------------------------------------------------------------------
    # Posterior quantities
    # ------------------------------------------------------------------

    def posterior_mean(self, team: str) -> dict[str, float]:
        """Posterior E[attack] and E[defense] with standard deviations."""
        a_att, b_att = self._att.get(team, (self._PRIOR_ALPHA, self._PRIOR_BETA))
        a_def, b_def = self._def.get(team, (self._PRIOR_ALPHA, self._PRIOR_BETA))
        return {
            "attack_mean": a_att / b_att,
            "defense_mean": a_def / b_def,
            "attack_std": math.sqrt(a_att) / b_att,
            "defense_std": math.sqrt(a_def) / b_def,
            "attack_alpha": a_att,
            "attack_beta": b_att,
            "defense_alpha": a_def,
            "defense_beta": b_def,
            "n_matches_implied": max(0.0, b_att - self._PRIOR_BETA),
        }

    def credible_interval(
        self,
        team: str,
        kind: str = "attack",
        ci: float = 0.90,
    ) -> tuple[float, float]:
        """
        Approximate (ci×100)% credible interval for attack or defense rate
        using the Gamma distribution and a normal approximation for tails.
        """
        if kind == "attack":
            a, b = self._att.get(team, (self._PRIOR_ALPHA, self._PRIOR_BETA))
        else:
            a, b = self._def.get(team, (self._PRIOR_ALPHA, self._PRIOR_BETA))

        alpha_tail = (1.0 - ci) / 2.0
        z_lo = _normal_quantile(alpha_tail)
        z_hi = _normal_quantile(1.0 - alpha_tail)
        mean = a / b
        cv = 1.0 / math.sqrt(a)  # Gamma coefficient of variation
        lo = max(mean * (1.0 + z_lo * cv), 0.0)
        hi = mean * (1.0 + z_hi * cv)
        return lo, hi

    # ------------------------------------------------------------------
    # Posterior predictive PMF
    # ------------------------------------------------------------------

    def _goal_pmf(self, att_team: str, def_team: str, mu_venue: float) -> list[float]:
        """
        Posterior predictive PMF for goals scored by att_team vs def_team.

        Derivation:
          E[λ_att] = α_att / β_att
          E[λ_def] = α_def / β_def  (measures defensive weakness)
          μ_eff    = mu_venue × E[λ_att] × E[λ_def]

          Marginalising λ_att ~ Gamma(α_att, β_att):
            goals ~ NegBin(r=α_att,  p = μ_eff / (α_att + μ_eff))
        """
        a_att, b_att = self._att.get(att_team, (self._PRIOR_ALPHA, self._PRIOR_BETA))
        a_def, b_def = self._def.get(def_team, (self._PRIOR_ALPHA, self._PRIOR_BETA))

        mu_att = a_att / b_att
        mu_def = a_def / b_def
        mu_eff = max(mu_venue * mu_att * mu_def, 1e-6)

        r = a_att
        p = mu_eff / (r + mu_eff)
        p = min(max(p, 1e-9), 1.0 - 1e-9)

        pmf = [_neg_binom_pmf(k, r, p) for k in range(self._MAX_GOALS + 1)]
        total = sum(pmf)
        return [v / total for v in pmf] if total > 0.0 else pmf

    # ------------------------------------------------------------------
    # Public prediction API
    # ------------------------------------------------------------------

    def predict_teams(self, home_team: str, away_team: str) -> dict[str, float]:
        """Full Bayesian posterior predictive 1×2 probabilities."""
        pmf_h = self._goal_pmf(home_team, away_team, self._league_mu_home)
        pmf_a = self._goal_pmf(away_team, home_team, self._league_mu_away)
        return self._joint_1x2(pmf_h, pmf_a)

    def predict(
        self,
        home_attack: float,
        home_defense: float,
        away_attack: float,
        away_defense: float,
    ) -> dict[str, float]:
        """
        Legacy strength-ratio API (backwards compatible).

        Converts raw strength ratios into Negative Binomial posterior
        predictive PMFs by constructing synthetic Gamma posteriors
        consistent with one observed match at the given rate.
        """

        def _pmf(att_rate: float, def_rate: float, mu_venue: float) -> list[float]:
            a_att = self._PRIOR_ALPHA + att_rate
            b_att = self._PRIOR_BETA + 1.0
            a_def = self._PRIOR_ALPHA + def_rate
            b_def = self._PRIOR_BETA + 1.0
            mu_eff = max(mu_venue * (a_att / b_att) * (a_def / b_def), 1e-6)
            r = a_att
            p = mu_eff / (r + mu_eff)
            p = min(max(p, 1e-9), 1.0 - 1e-9)
            raw = [_neg_binom_pmf(k, r, p) for k in range(self._MAX_GOALS + 1)]
            total = sum(raw)
            return [v / total for v in raw] if total > 0.0 else raw

        pmf_h = _pmf(home_attack, away_defense, self._league_mu_home)
        pmf_a = _pmf(away_attack, home_defense, self._league_mu_away)
        return self._joint_1x2(pmf_h, pmf_a)

    def predict_score_distribution(
        self, home_team: str, away_team: str
    ) -> dict[str, list[float]]:
        """Return full PMF vectors for scoreline pricing and O/U markets."""
        return {
            "home_goals_pmf": self._goal_pmf(
                home_team, away_team, self._league_mu_home
            ),
            "away_goals_pmf": self._goal_pmf(
                away_team, home_team, self._league_mu_away
            ),
        }

    def log_score(
        self,
        home_team: str,
        away_team: str,
        actual_home: int,
        actual_away: int,
    ) -> float:
        """Log-likelihood of the actual scoreline under the posterior predictive."""
        pmf_h = self._goal_pmf(home_team, away_team, self._league_mu_home)
        pmf_a = self._goal_pmf(away_team, home_team, self._league_mu_away)
        ph = pmf_h[actual_home] if actual_home <= self._MAX_GOALS else 0.0
        pa = pmf_a[actual_away] if actual_away <= self._MAX_GOALS else 0.0
        return math.log(max(ph * pa, 1e-15))

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _joint_1x2(pmf_h: list[float], pmf_a: list[float]) -> dict[str, float]:
        home_win = draw = away_win = 0.0
        for hg, ph in enumerate(pmf_h):
            for ag, pa in enumerate(pmf_a):
                p = ph * pa
                if hg > ag:
                    home_win += p
                elif hg == ag:
                    draw += p
                else:
                    away_win += p
        total = home_win + draw + away_win
        if total <= 0.0:
            return {"home_win": 1.0 / 3, "draw": 1.0 / 3, "away_win": 1.0 / 3}
        return {
            "home_win": home_win / total,
            "draw": draw / total,
            "away_win": away_win / total,
        }

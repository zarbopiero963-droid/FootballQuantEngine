"""
Halftime Model
==============
Predicts Over/Under for the first half and second half independently.

Approach:
  - Fit separate average halftime goal rates from historical ht_home / ht_away data.
  - HT lambda ≈ 40-45 % of full-time lambda (empirical ratio calibrated from data).
  - Second half lambda = full-time lambda − halftime lambda.
  - O/U computed from independent Poisson PMFs convolved for total half-goals.
"""

from __future__ import annotations

import math


class HalftimeModel:

    # Fallback ratios when no halftime data is available
    _DEFAULT_HT_RATIO = 0.42

    def __init__(self, max_goals: int = 8):
        self.max_goals = max_goals
        self._ht_ratio_home = self._DEFAULT_HT_RATIO
        self._ht_ratio_away = self._DEFAULT_HT_RATIO

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, completed_matches: list[dict]) -> None:
        """
        Calibrate the first-half ratio from historical data.
        Each match dict must have: home_goals, away_goals, ht_home, ht_away.
        """
        valid = [
            m
            for m in completed_matches
            if m.get("ht_home") is not None
            and m.get("ht_away") is not None
            and m.get("home_goals") is not None
            and m.get("away_goals") is not None
            and int(m["home_goals"]) > 0
            and int(m["away_goals"]) > 0
        ]

        if len(valid) < 20:
            return  # Not enough data — keep defaults

        ht_home_sum = sum(float(m["ht_home"]) for m in valid)
        ft_home_sum = sum(float(m["home_goals"]) for m in valid)
        ht_away_sum = sum(float(m["ht_away"]) for m in valid)
        ft_away_sum = sum(float(m["away_goals"]) for m in valid)

        self._ht_ratio_home = (
            ht_home_sum / ft_home_sum if ft_home_sum else self._DEFAULT_HT_RATIO
        )
        self._ht_ratio_away = (
            ht_away_sum / ft_away_sum if ft_away_sum else self._DEFAULT_HT_RATIO
        )

        # Clamp to sensible range
        self._ht_ratio_home = max(0.30, min(0.55, self._ht_ratio_home))
        self._ht_ratio_away = max(0.30, min(0.55, self._ht_ratio_away))

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def probabilities(
        self,
        lambda_home_ft: float,
        lambda_away_ft: float,
        line_ht: float = 0.5,
        line_sh: float = 0.5,
    ) -> dict:
        """
        Returns O/U probabilities for both halves.

        Args:
            lambda_home_ft: expected home goals for full match
            lambda_away_ft: expected away goals for full match
            line_ht: over/under line for the first half (default 0.5)
            line_sh: over/under line for the second half (default 0.5)
        """
        lam_ht_home = lambda_home_ft * self._ht_ratio_home
        lam_ht_away = lambda_away_ft * self._ht_ratio_away
        lam_sh_home = max(0.01, lambda_home_ft - lam_ht_home)
        lam_sh_away = max(0.01, lambda_away_ft - lam_ht_away)

        ht_ou = self._ou_total(lam_ht_home, lam_ht_away, line_ht)
        sh_ou = self._ou_total(lam_sh_home, lam_sh_away, line_sh)

        return {
            "ht": {
                "lambda_home": round(lam_ht_home, 3),
                "lambda_away": round(lam_ht_away, 3),
                "lambda_total": round(lam_ht_home + lam_ht_away, 3),
                "line": line_ht,
                "over": round(ht_ou["over"], 4),
                "under": round(ht_ou["under"], 4),
            },
            "sh": {
                "lambda_home": round(lam_sh_home, 3),
                "lambda_away": round(lam_sh_away, 3),
                "lambda_total": round(lam_sh_home + lam_sh_away, 3),
                "line": line_sh,
                "over": round(sh_ou["over"], 4),
                "under": round(sh_ou["under"], 4),
            },
        }

    def all_lines(self, lambda_home_ft: float, lambda_away_ft: float) -> dict:
        """Standard lines for both halves."""
        result = {}
        for line in [0.5, 1.5]:
            probs = self.probabilities(
                lambda_home_ft, lambda_away_ft, line_ht=line, line_sh=line
            )
            result[f"ht_ou{str(line).replace('.', '')}"] = probs["ht"]
            result[f"sh_ou{str(line).replace('.', '')}"] = probs["sh"]
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _poisson_pmf(self, k: int, lam: float) -> float:
        lam = max(0.01, lam)
        return math.exp(-lam) * (lam**k) / math.factorial(min(k, 170))

    def _ou_total(self, lam_h: float, lam_a: float, line: float) -> dict:
        pmf_h = [self._poisson_pmf(k, lam_h) for k in range(self.max_goals + 1)]
        pmf_a = [self._poisson_pmf(k, lam_a) for k in range(self.max_goals + 1)]
        norm_h = sum(pmf_h) or 1.0
        norm_a = sum(pmf_a) or 1.0
        pmf_h = [p / norm_h for p in pmf_h]
        pmf_a = [p / norm_a for p in pmf_a]

        over = under = 0.0
        for h in range(self.max_goals + 1):
            for a in range(self.max_goals + 1):
                prob = pmf_h[h] * pmf_a[a]
                if h + a > line:
                    over += prob
                else:
                    under += prob

        total = over + under or 1.0
        return {"over": over / total, "under": under / total}

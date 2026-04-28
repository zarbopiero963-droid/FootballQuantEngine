"""
Corners Model
=============
Poisson-based model for corner kicks.

Approach:
  - Each team has an attack_corners and defense_corners strength (same
    Dixon-Coles normalisation used for goals).
  - λ_home_corners = league_avg_home_corners × home_attack_corners × away_defense_corners
  - λ_away_corners = league_avg_away_corners × away_attack_corners × home_defense_corners
  - Total corners = Poisson(λ_home) + Poisson(λ_away)  (independent)
  - P(total > line) computed by convolution of the two PMFs.
"""

from __future__ import annotations

import math


class CornersModel:

    def __init__(self, max_corners: int = 20):
        self.max_corners = max_corners
        self.league_avg_home_corners = 5.5
        self.league_avg_away_corners = 4.5
        self._team_stats: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, completed_matches: list[dict]) -> None:
        """
        completed_matches: list of dicts with keys
            home_team, away_team, home_corners, away_corners
        Rows with missing corner data are skipped.
        """
        valid = [
            m
            for m in completed_matches
            if m.get("home_corners") is not None and m.get("away_corners") is not None
        ]
        if not valid:
            return

        total_home = sum(float(m["home_corners"]) for m in valid)
        total_away = sum(float(m["away_corners"]) for m in valid)
        n = len(valid)
        self.league_avg_home_corners = total_home / n
        self.league_avg_away_corners = total_away / n

        raw: dict[str, dict] = {}
        for m in valid:
            home = m["home_team"]
            away = m["away_team"]
            hc = float(m["home_corners"])
            ac = float(m["away_corners"])

            for team in (home, away):
                if team not in raw:
                    raw[team] = {
                        "home_won": 0.0,
                        "home_conceded": 0.0,
                        "home_n": 0,
                        "away_won": 0.0,
                        "away_conceded": 0.0,
                        "away_n": 0,
                    }

            raw[home]["home_won"] += hc
            raw[home]["home_conceded"] += ac
            raw[home]["home_n"] += 1
            raw[away]["away_won"] += ac
            raw[away]["away_conceded"] += hc
            raw[away]["away_n"] += 1

        fitted: dict[str, dict] = {}
        for team, v in raw.items():
            hn = max(v["home_n"], 1)
            an = max(v["away_n"], 1)
            avg_hw = v["home_won"] / hn
            avg_hc = v["home_conceded"] / hn
            avg_aw = v["away_won"] / an
            avg_ac = v["away_conceded"] / an

            fitted[team] = {
                "attack_home": (
                    avg_hw / self.league_avg_home_corners
                    if self.league_avg_home_corners
                    else 1.0
                ),
                "defense_home": (
                    avg_hc / self.league_avg_away_corners
                    if self.league_avg_away_corners
                    else 1.0
                ),
                "attack_away": (
                    avg_aw / self.league_avg_away_corners
                    if self.league_avg_away_corners
                    else 1.0
                ),
                "defense_away": (
                    avg_ac / self.league_avg_home_corners
                    if self.league_avg_home_corners
                    else 1.0
                ),
            }

        self._team_stats = fitted

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def expected_corners(self, home_team: str, away_team: str) -> tuple[float, float]:
        home_p = self._team_stats.get(
            home_team,
            {
                "attack_home": 1.0,
                "defense_home": 1.0,
                "attack_away": 1.0,
                "defense_away": 1.0,
            },
        )
        away_p = self._team_stats.get(
            away_team,
            {
                "attack_home": 1.0,
                "defense_home": 1.0,
                "attack_away": 1.0,
                "defense_away": 1.0,
            },
        )

        lam_home = max(
            0.5,
            self.league_avg_home_corners
            * home_p["attack_home"]
            * away_p["defense_away"],
        )
        lam_away = max(
            0.5,
            self.league_avg_away_corners
            * away_p["attack_away"]
            * home_p["defense_home"],
        )
        return lam_home, lam_away

    def probabilities(self, home_team: str, away_team: str, line: float = 9.5) -> dict:
        lam_h, lam_a = self.expected_corners(home_team, away_team)

        # Build total-corners PMF by convolution
        pmf_h = self._poisson_pmf_vec(lam_h)
        pmf_a = self._poisson_pmf_vec(lam_a)

        # Convolve: P(total = k)
        max_total = 2 * self.max_corners
        pmf_total = [0.0] * (max_total + 1)
        for h in range(self.max_corners + 1):
            for a in range(self.max_corners + 1):
                if h + a <= max_total:
                    pmf_total[h + a] += pmf_h[h] * pmf_a[a]

        over = sum(p for k, p in enumerate(pmf_total) if k > line)
        under = sum(p for k, p in enumerate(pmf_total) if k <= line)
        total = over + under or 1.0

        lam_total = lam_h + lam_a
        return {
            "lambda_home": round(lam_h, 3),
            "lambda_away": round(lam_a, 3),
            "lambda_total": round(lam_total, 3),
            "line": line,
            "over": round(over / total, 4),
            "under": round(under / total, 4),
        }

    def all_lines(self, home_team: str, away_team: str) -> dict:
        """Return over/under probabilities for the three most common corner lines."""
        result = {}
        for line in [8.5, 9.5, 10.5]:
            result[f"corners_{line}"] = self.probabilities(home_team, away_team, line)
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _poisson_pmf(self, k: int, lam: float) -> float:
        lam = max(0.01, lam)
        return math.exp(-lam) * (lam**k) / math.factorial(min(k, 170))

    def _poisson_pmf_vec(self, lam: float) -> list[float]:
        raw = [self._poisson_pmf(k, lam) for k in range(self.max_corners + 1)]
        total = sum(raw) or 1.0
        return [p / total for p in raw]

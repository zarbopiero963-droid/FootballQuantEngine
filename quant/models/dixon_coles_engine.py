from __future__ import annotations

import math

from quant.models.poisson_engine import PoissonEngine

_RHO_CANDIDATES = [r / 100.0 for r in range(-25, 26, 5)]


class DixonColesEngine(PoissonEngine):
    """
    Extends Poisson with Dixon-Coles correlation correction.
    rho adjusts the joint probability for low-scoring scorelines
    (0-0, 1-0, 0-1, 1-1) to improve draw prediction.
    """

    def __init__(self, max_goals: int = 8):
        super().__init__(max_goals=max_goals)
        self.rho: float = 0.0

    def _tau(self, hg: int, ag: int, lh: float, la: float) -> float:
        if hg == 0 and ag == 0:
            return 1.0 - lh * la * self.rho
        if hg == 0 and ag == 1:
            return 1.0 + lh * self.rho
        if hg == 1 and ag == 0:
            return 1.0 + la * self.rho
        if hg == 1 and ag == 1:
            return 1.0 - self.rho
        return 1.0

    def _log_likelihood(self, completed_matches: list[dict], rho: float) -> float:
        self.rho = rho
        ll = 0.0
        for match in completed_matches:
            lh, la = self.expected_goals(match["home_team"], match["away_team"])
            hg = int(match["home_goals"])
            ag = int(match["away_goals"])
            tau = self._tau(hg, ag, lh, la)
            p_h = math.exp(-lh) * (lh**hg) / math.factorial(min(hg, 20))
            p_a = math.exp(-la) * (la**ag) / math.factorial(min(ag, 20))
            p = p_h * p_a * tau
            if p > 1e-12:
                ll += math.log(p)
        return ll

    def _estimate_rho(self, completed_matches: list[dict]) -> float:
        best_rho, best_ll = 0.0, float("-inf")
        for candidate in _RHO_CANDIDATES:
            ll = self._log_likelihood(completed_matches, candidate)
            if ll > best_ll:
                best_ll, best_rho = ll, candidate
        return best_rho

    def fit(self, completed_matches: list[dict]) -> None:
        super().fit(completed_matches)
        if completed_matches:
            self.rho = self._estimate_rho(completed_matches)

    def probabilities_1x2_from_lambdas(self, lh: float, la: float) -> dict:
        """Compute DC-corrected 1x2 probabilities directly from lambda values."""
        home_win = draw = away_win = 0.0
        for hg in range(self.max_goals + 1):
            for ag in range(self.max_goals + 1):
                p_h = math.exp(-lh) * (lh**hg) / math.factorial(hg)
                p_a = math.exp(-la) * (la**ag) / math.factorial(ag)
                p = p_h * p_a * self._tau(hg, ag, lh, la)
                if hg > ag:
                    home_win += p
                elif hg == ag:
                    draw += p
                else:
                    away_win += p
        total = home_win + draw + away_win
        if total <= 0:
            return {"home_win": 1 / 3, "draw": 1 / 3, "away_win": 1 / 3}
        return {
            "home_win": home_win / total,
            "draw": draw / total,
            "away_win": away_win / total,
        }

    def probabilities_1x2(self, home_team: str, away_team: str) -> dict:
        lh, la = self.expected_goals(home_team, away_team)
        home_win = draw = away_win = 0.0

        for hg in range(self.max_goals + 1):
            for ag in range(self.max_goals + 1):
                p_h = math.exp(-lh) * (lh**hg) / math.factorial(hg)
                p_a = math.exp(-la) * (la**ag) / math.factorial(ag)
                p = p_h * p_a * self._tau(hg, ag, lh, la)
                if hg > ag:
                    home_win += p
                elif hg == ag:
                    draw += p
                else:
                    away_win += p

        total = home_win + draw + away_win
        if total <= 0:
            return {"home_win": 1 / 3, "draw": 1 / 3, "away_win": 1 / 3}
        return {
            "home_win": home_win / total,
            "draw": draw / total,
            "away_win": away_win / total,
        }

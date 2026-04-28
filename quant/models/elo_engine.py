"""
ELO rating engine with per-league K-factor and home-advantage calibration.

Default values (K=20, home_adv=55) are reasonable starting points but were not
derived from data.  Use calibrate() to grid-search optimal values on a
held-out validation split of completed matches, minimising Brier score of the
ELO expected-score prediction vs. the actual match result.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Grid candidates for calibration
_K_CANDIDATES = [10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0]
_HOME_ADV_CANDIDATES = [30.0, 40.0, 50.0, 55.0, 60.0, 70.0, 80.0]


class EloEngine:

    def __init__(self, base_rating: float = 1500.0, k_factor: float = 20.0,
                 home_advantage: float = 55.0):
        self.base_rating = base_rating
        self.k_factor = k_factor
        self.home_advantage = home_advantage
        self.ratings: dict = {}

    def get_rating(self, team_name: str) -> float:
        return self.ratings.get(team_name, self.base_rating)

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))

    def update_match(self, home_team: str, away_team: str,
                     home_goals: int, away_goals: int) -> None:
        home_rating = self.get_rating(home_team)
        away_rating = self.get_rating(away_team)

        adjusted_home = home_rating + self.home_advantage
        expected_home = self.expected_score(adjusted_home, away_rating)
        expected_away = 1.0 - expected_home

        if home_goals > away_goals:
            actual_home, actual_away = 1.0, 0.0
        elif home_goals < away_goals:
            actual_home, actual_away = 0.0, 1.0
        else:
            actual_home, actual_away = 0.5, 0.5

        self.ratings[home_team] = home_rating + self.k_factor * (actual_home - expected_home)
        self.ratings[away_team] = away_rating + self.k_factor * (actual_away - expected_away)

    def fit(self, completed_matches: list) -> None:
        for match in completed_matches:
            self.update_match(
                home_team=match["home_team"],
                away_team=match["away_team"],
                home_goals=int(match["home_goals"]),
                away_goals=int(match["away_goals"]),
            )

    def get_elo_diff(self, home_team: str, away_team: str) -> float:
        home_rating = self.get_rating(home_team)
        away_rating = self.get_rating(away_team)
        return (home_rating + self.home_advantage) - away_rating

    # ------------------------------------------------------------------
    # Calibration via backtest grid search
    # ------------------------------------------------------------------

    def calibrate(
        self,
        completed_matches: list,
        train_ratio: float = 0.7,
        k_candidates: Optional[list] = None,
        home_adv_candidates: Optional[list] = None,
    ) -> dict:
        """
        Grid-search optimal K-factor and home_advantage on *completed_matches*.

        Strategy
        --------
        1. Split chronologically: first ``train_ratio`` of matches for warm-up,
           remainder for evaluation.
        2. For each (K, home_adv) candidate pair:
           a. Fit ELO on the training split.
           b. Evaluate Brier score (E_home − actual_home)² on the eval split,
              updating ratings as each eval match is processed.
        3. Store and apply the best (K, home_adv).

        Returns
        -------
        dict with keys: k_factor, home_advantage, brier_score, n_eval
        """
        if not completed_matches:
            logger.warning("EloEngine.calibrate called with empty match list; using defaults")
            return {"k_factor": self.k_factor, "home_advantage": self.home_advantage,
                    "brier_score": None, "n_eval": 0}

        k_grid = k_candidates or _K_CANDIDATES
        ha_grid = home_adv_candidates or _HOME_ADV_CANDIDATES

        n = len(completed_matches)
        n_train = max(1, int(n * train_ratio))
        train = completed_matches[:n_train]
        eval_ = completed_matches[n_train:]

        if not eval_:
            logger.warning("EloEngine.calibrate: not enough matches for eval split; using defaults")
            return {"k_factor": self.k_factor, "home_advantage": self.home_advantage,
                    "brier_score": None, "n_eval": 0}

        best_brier = float("inf")
        best_k = self.k_factor
        best_ha = self.home_advantage

        for k in k_grid:
            for ha in ha_grid:
                engine = EloEngine(base_rating=self.base_rating, k_factor=k,
                                   home_advantage=ha)
                engine.fit(train)

                brier_sum = 0.0
                for match in eval_:
                    home_r = engine.get_rating(match["home_team"])
                    away_r = engine.get_rating(match["away_team"])
                    exp_home = engine.expected_score(home_r + ha, away_r)

                    hg = int(match["home_goals"])
                    ag = int(match["away_goals"])
                    if hg > ag:
                        actual = 1.0
                    elif hg < ag:
                        actual = 0.0
                    else:
                        actual = 0.5

                    brier_sum += (exp_home - actual) ** 2
                    engine.update_match(match["home_team"], match["away_team"], hg, ag)

                brier = brier_sum / len(eval_)
                if brier < best_brier:
                    best_brier = brier
                    best_k = k
                    best_ha = ha

        self.k_factor = best_k
        self.home_advantage = best_ha
        logger.info(
            "EloEngine calibration: K=%.1f home_adv=%.1f Brier=%.5f (n_eval=%d)",
            best_k, best_ha, best_brier, len(eval_),
        )
        return {
            "k_factor": best_k,
            "home_advantage": best_ha,
            "brier_score": round(best_brier, 6),
            "n_eval": len(eval_),
        }

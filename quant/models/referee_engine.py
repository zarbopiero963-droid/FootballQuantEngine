from __future__ import annotations


class RefereeEngine:
    """
    Referee tendency profile: home bias and strictness.
    Primarily influences expected game flow (cards, penalties).
    Applied as a small lambda modifier.
    """

    def __init__(self):
        self._stats: dict[str, dict] = {}

    def fit(self, referee_stats: dict[str, dict] | None) -> None:
        """
        referee_stats: {
            "Referee Name": {
                "home_bias": float,   # positive = favors home
                "strictness": float,  # 1.0 = league average cards
                "games": int,
            }
        }
        """
        self._stats = referee_stats or {}

    def get_home_bias(self, referee_name: str | None) -> float:
        if not referee_name:
            return 0.0
        return float(self._stats.get(referee_name, {}).get("home_bias", 0.0))

    def get_strictness(self, referee_name: str | None) -> float:
        if not referee_name:
            return 1.0
        return float(self._stats.get(referee_name, {}).get("strictness", 1.0))

    def get_lambda_modifiers(self, referee_name: str | None) -> tuple[float, float]:
        """
        Returns (home_modifier, away_modifier).
        A positive home_bias slightly boosts home expected goals.
        """
        bias = self.get_home_bias(referee_name)
        home_mod = 1.0 + bias * 0.05
        away_mod = 1.0 - bias * 0.05
        return home_mod, away_mod

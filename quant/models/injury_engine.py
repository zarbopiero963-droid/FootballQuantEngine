from __future__ import annotations

_POSITION_WEIGHTS: dict[str, float] = {
    "Goalkeeper": 1.2,
    "Defender": 0.8,
    "Midfielder": 0.7,
    "Attacker": 0.9,
    "Forward": 0.9,
}


class InjuryEngine:
    """
    Estimates the performance impact of player absences per team.
    Returns a factor in [0.70, 1.0]: lower = more impacted.
    """

    def __init__(self):
        self._injuries: dict[str, list[dict]] = {}

    def fit(self, injuries_data: dict[str, list[dict]] | None) -> None:
        self._injuries = injuries_data or {}

    def get_injury_factor(self, team: str) -> float:
        absences = self._injuries.get(team, [])
        if not absences:
            return 1.0
        total_impact = sum(
            _POSITION_WEIGHTS.get(inj.get("position", "Midfielder"), 0.7)
            for inj in absences
        )
        return max(0.70, 1.0 - total_impact * 0.05)

    def get_injury_diff(self, home_team: str, away_team: str) -> float:
        """Positive = home team less impacted by injuries."""
        return self.get_injury_factor(home_team) - self.get_injury_factor(away_team)

    def get_lambda_modifiers(
        self, home_team: str, away_team: str
    ) -> tuple[float, float]:
        """Returns (home_modifier, away_modifier) to multiply expected goals."""
        return self.get_injury_factor(home_team), self.get_injury_factor(away_team)

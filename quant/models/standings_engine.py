from __future__ import annotations


class StandingsEngine:
    """
    Derives motivation factor from league table position.
    Teams in title race (top 4) or relegation battle (bottom 3) have
    higher stake → treated as having elevated motivation.
    """

    def __init__(self):
        self._standings: dict[str, dict] = {}
        self._total_teams: int = 20

    def fit(self, standings_data: list[dict] | dict | None) -> None:
        if not standings_data:
            return
        if isinstance(standings_data, list):
            for row in standings_data:
                team = row.get("team")
                if team:
                    self._standings[team] = row
            self._total_teams = max(len(standings_data), 1)
        elif isinstance(standings_data, dict):
            self._standings = standings_data
            self._total_teams = max(len(standings_data), 1)

    def _motivation(self, team: str) -> float:
        if team not in self._standings:
            return 0.5
        pos = int(self._standings[team].get("position", self._total_teams // 2))
        # Title zone
        if pos <= 4:
            return min(1.0, 0.80 + (4 - pos) * 0.05)
        # Relegation zone (bottom 3)
        if pos >= self._total_teams - 2:
            return min(1.0, 0.75 + (self._total_teams - pos) * 0.05)
        return 0.5

    def get_motivation_diff(self, home_team: str, away_team: str) -> float:
        """Positive = home team more motivated."""
        return self._motivation(home_team) - self._motivation(away_team)

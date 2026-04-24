from __future__ import annotations

from datetime import datetime, timezone


class RestEngine:
    """
    Computes rest days between a team's last match and the upcoming fixture.
    Fewer rest days → fatigue factor that reduces effective attack strength.
    """

    _FATIGUE: list[tuple[int, float]] = [
        (2, 0.85),
        (3, 0.92),
        (5, 0.97),
    ]

    def __init__(self):
        self._last_match: dict[str, datetime] = {}

    def fit(self, completed_matches: list[dict]) -> None:
        for match in completed_matches:
            date_str = match.get("match_date") or match.get("date")
            if not date_str:
                continue
            try:
                dt = datetime.fromisoformat(str(date_str).replace("Z", "+00:00"))
            except (ValueError, TypeError):
                continue
            for key in ("home_team", "away_team"):
                team = match.get(key)
                if team and (
                    team not in self._last_match or dt > self._last_match[team]
                ):
                    self._last_match[team] = dt

    def get_rest_days(self, team: str, fixture_dt: datetime | None = None) -> int:
        if team not in self._last_match:
            return 7
        ref = fixture_dt or datetime.now(timezone.utc)
        if ref.tzinfo is None:
            ref = ref.replace(tzinfo=timezone.utc)
        last = self._last_match[team]
        if last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)
        return max(0, (ref - last).days)

    def get_fatigue_factor(self, rest_days: int) -> float:
        for threshold, factor in self._FATIGUE:
            if rest_days <= threshold:
                return factor
        return 1.0

    def get_rest_diff(
        self, home_team: str, away_team: str, fixture_dt: datetime | None = None
    ) -> float:
        """Positive = home team is better rested."""
        return self.get_rest_days(home_team, fixture_dt) - self.get_rest_days(
            away_team, fixture_dt
        )

    def get_lambda_modifiers(
        self, home_team: str, away_team: str, fixture_dt: datetime | None = None
    ) -> tuple[float, float]:
        """Returns (home_modifier, away_modifier) to multiply expected goals."""
        home_mod = self.get_fatigue_factor(self.get_rest_days(home_team, fixture_dt))
        away_mod = self.get_fatigue_factor(self.get_rest_days(away_team, fixture_dt))
        return home_mod, away_mod

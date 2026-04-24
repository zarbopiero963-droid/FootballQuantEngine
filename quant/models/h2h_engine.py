from __future__ import annotations


class H2HEngine:
    """
    Head-to-head dominance from historical matchups between two teams.
    Returns a score in [-1, 1]: positive = home historically dominant.
    """

    def __init__(self, lookback: int = 10):
        self.lookback = lookback
        self._history: dict[tuple, list[float]] = {}

    def fit(self, completed_matches: list[dict]) -> None:
        history: dict[tuple, list[float]] = {}
        for match in completed_matches:
            home = match["home_team"]
            away = match["away_team"]
            hg = int(match["home_goals"])
            ag = int(match["away_goals"])
            key = (home, away)
            history.setdefault(key, [])
            if hg > ag:
                history[key].append(1.0)
            elif hg == ag:
                history[key].append(0.5)
            else:
                history[key].append(0.0)
        self._history = history

    def get_h2h_diff(self, home_team: str, away_team: str) -> float:
        """Positive = home team historically dominates this fixture."""
        direct = self._history.get((home_team, away_team), [])
        reversed_ = self._history.get((away_team, home_team), [])

        results: list[float] = []
        for r in direct[-self.lookback :]:
            results.append(r)
        for r in reversed_[-self.lookback :]:
            results.append(1.0 - r)

        if not results:
            return 0.0

        avg = sum(results) / len(results)
        return (avg - 0.5) * 2.0  # scale [-1, 1]

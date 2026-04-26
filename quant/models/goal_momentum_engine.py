from __future__ import annotations


class GoalMomentumEngine:
    """
    Exponentially-weighted goal-scoring momentum over the last N matches.
    More recent matches carry higher weight.
    """

    def __init__(self, lookback: int = 5, decay: float = 0.75):
        self.lookback = lookback
        self.decay = decay
        self._momentum: dict[str, float] = {}

    def fit(self, completed_matches: list[dict]) -> None:
        history: dict[str, list[float]] = {}
        for match in completed_matches:
            home = match["home_team"]
            away = match["away_team"]
            hg = float(match["home_goals"])
            ag = float(match["away_goals"])
            history.setdefault(home, []).append(hg)
            history.setdefault(away, []).append(ag)

        momentum: dict[str, float] = {}
        for team, goals in history.items():
            recent = goals[-self.lookback :]
            if not recent:
                momentum[team] = 1.2
                continue
            weights = [self.decay ** (len(recent) - 1 - i) for i in range(len(recent))]
            w_sum = sum(weights)
            momentum[team] = sum(g * w for g, w in zip(recent, weights)) / w_sum
        self._momentum = momentum

    def get_momentum(self, team: str) -> float:
        return self._momentum.get(team, 1.2)

    def get_momentum_diff(self, home_team: str, away_team: str) -> float:
        """Positive = home team scoring more in recent matches."""
        return self.get_momentum(home_team) - self.get_momentum(away_team)

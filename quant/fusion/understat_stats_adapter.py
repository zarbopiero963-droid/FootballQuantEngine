from __future__ import annotations

from quant.fusion.team_mapper import TeamMapper
from quant.fusion.team_name_normalizer import TeamNameNormalizer


class UnderstatStatsAdapter:

    def __init__(self, mapper: TeamMapper | None = None):
        self.mapper = mapper or TeamMapper()
        self.normalizer = TeamNameNormalizer()

    def adapt(self, understat_stats: dict | None) -> dict:
        understat_stats = understat_stats or {}
        adapted = {}

        for team_name, payload in understat_stats.items():
            key = self.mapper.map_name(team_name)
            adapted[key] = {
                "source_team_name": team_name,
                "mapped_team_name": key,
                "matches": int(payload.get("matches", 0) or 0),
                "xg_for": float(payload.get("xg_for", 1.2) or 1.2),
                "xg_against": float(payload.get("xg_against", 1.2) or 1.2),
                "xpts": float(payload.get("xpts", 1.4) or 1.4),
                "goals_for": float(payload.get("goals_for", 1.2) or 1.2),
                "goals_against": float(payload.get("goals_against", 1.2) or 1.2),
                "shots": float(payload.get("shots", 10.0) or 10.0),
                "shots_on_target": float(payload.get("shots_on_target", 3.5) or 3.5),
                "deep": float(payload.get("deep", 8.0) or 8.0),
                "ppda_att": float(payload.get("ppda_att", 10.0) or 10.0),
                "ppda_def": float(payload.get("ppda_def", 10.0) or 10.0),
                "xg_diff": float(payload.get("xg_diff", 0.0) or 0.0),
                "goal_diff": float(payload.get("goal_diff", 0.0) or 0.0),
                "finishing_delta": float(payload.get("finishing_delta", 0.0) or 0.0),
                "defensive_delta": float(payload.get("defensive_delta", 0.0) or 0.0),
                "form_score": float(payload.get("form_score", 0.5) or 0.5),
            }

        return adapted

    def get_team_stats(self, adapted_stats: dict, team_name: str) -> dict:
        key = self.mapper.map_name(team_name)

        if key in adapted_stats:
            return adapted_stats[key]

        return {
            "source_team_name": team_name,
            "mapped_team_name": key,
            "matches": 0,
            "xg_for": 1.2,
            "xg_against": 1.2,
            "xpts": 1.4,
            "goals_for": 1.2,
            "goals_against": 1.2,
            "shots": 10.0,
            "shots_on_target": 3.5,
            "deep": 8.0,
            "ppda_att": 10.0,
            "ppda_def": 10.0,
            "xg_diff": 0.0,
            "goal_diff": 0.0,
            "finishing_delta": 0.0,
            "defensive_delta": 0.0,
            "form_score": 0.5,
        }

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TeamStrength:
    team_name: str
    elo: float = 1500.0
    attack_home: float = 1.0
    defense_home: float = 1.0
    attack_away: float = 1.0
    defense_away: float = 1.0
    xg_for: float = 1.0
    xg_against: float = 1.0
    form_score: float = 0.0

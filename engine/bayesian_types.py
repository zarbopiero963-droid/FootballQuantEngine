"""
Data types and constants for the Bayesian Live Engine.

Extracted from bayesian_live.py to keep each module under 400 LOC.
All public names are re-exported from engine.bayesian_live for backward compat.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EVENT_DELTA_ALPHA: Dict[str, Tuple[float, float]] = {
    "home_goal": (1.00, 0.00),
    "away_goal": (0.00, 1.00),
    "home_shot_on_target": (0.15, 0.00),
    "away_shot_on_target": (0.00, 0.15),
    "home_dangerous_attack": (0.05, 0.00),
    "away_dangerous_attack": (0.00, 0.05),
    "home_red_card": (0.00, 0.20),
    "away_red_card": (0.20, 0.00),
    "corner_home": (0.03, 0.00),
    "corner_away": (0.00, 0.03),
}

_MIN_ALPHA: float = 0.5
_MAX_GOALS: int = 10


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class PreMatchPrior:
    """Pre-match Poisson estimates used to build Gamma priors."""

    lambda_home: float
    lambda_away: float
    confidence: float = 3.0

    def __post_init__(self) -> None:
        if self.lambda_home <= 0 or self.lambda_away <= 0:
            raise ValueError("Pre-match lambda values must be positive.")
        if self.confidence <= 0:
            raise ValueError("Confidence must be positive.")


@dataclass
class LiveEvent:
    """A single in-play event observed during the match."""

    minute: int
    event_type: str
    timestamp: float = field(default_factory=lambda: time.time())

    def __post_init__(self) -> None:
        if self.event_type not in _EVENT_DELTA_ALPHA:
            raise ValueError(
                f"Unknown event_type '{self.event_type}'. "
                f"Valid types: {sorted(_EVENT_DELTA_ALPHA)}"
            )
        if not (0 <= self.minute <= 120):
            raise ValueError(f"minute must be in [0, 120], got {self.minute}.")


@dataclass
class BayesianState:
    """Full Bayesian posterior state at a given match minute."""

    minute: int
    home_score: int
    away_score: int
    alpha_home: float
    beta_home: float
    alpha_away: float
    beta_away: float
    posterior_lambda_home: float
    posterior_lambda_away: float
    p_home_win: float
    p_draw: float
    p_away_win: float
    p_over_25: float
    p_under_25: float
    p_btts: float
    events_processed: int
    last_event_minute: int

    def summary(self) -> str:
        sep = "=" * 56
        lines = [
            sep,
            f"  BAYESIAN LIVE STATE  —  minute {self.minute:>3d}",
            sep,
            f"  Score          : {self.home_score} – {self.away_score}",
            f"  λ home (post.) : {self.posterior_lambda_home:.4f}",
            f"  λ away (post.) : {self.posterior_lambda_away:.4f}",
            f"  Gamma home     : Gamma({self.alpha_home:.4f}, {self.beta_home:.4f})",
            f"  Gamma away     : Gamma({self.alpha_away:.4f}, {self.beta_away:.4f})",
            "  " + "-" * 52,
            f"  P(home win)    : {self.p_home_win:.4f}",
            f"  P(draw)        : {self.p_draw:.4f}",
            f"  P(away win)    : {self.p_away_win:.4f}",
            f"  P(over 2.5)    : {self.p_over_25:.4f}",
            f"  P(under 2.5)   : {self.p_under_25:.4f}",
            f"  P(BTTS)        : {self.p_btts:.4f}",
            "  " + "-" * 52,
            f"  Events done    : {self.events_processed}",
            f"  Last event min : {self.last_event_minute}",
            sep,
        ]
        return "\n".join(lines)


@dataclass
class BayesianAlert:
    """Triggered when the posterior diverges significantly from pre-match."""

    state: BayesianState
    alert_type: str
    message: str
    edge_direction: str
    confidence: str

    def __str__(self) -> str:
        return (
            f"[{self.confidence}] {self.alert_type}: {self.message} "
            f"→ {self.edge_direction}"
        )

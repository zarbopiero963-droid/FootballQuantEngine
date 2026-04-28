"""
Convenience runner for the Bayesian Live Engine.

Extracted from bayesian_live.py to keep that module under 400 LOC.
run_bayesian_live() is re-exported from engine.bayesian_live for backward compat.
"""

from __future__ import annotations

import time
from typing import List, Tuple

from engine.bayesian_types import BayesianState, LiveEvent, PreMatchPrior


def run_bayesian_live(
    pre_match_lambda_home: float,
    pre_match_lambda_away: float,
    events: List[dict],
    final_score: Tuple[int, int] = (0, 0),
    confidence: float = 3.0,
    quiet_window_minutes: int = 15,
    quiet_penalty: float = 0.08,
    alert_lambda_shift: float = 0.25,
) -> BayesianState:
    """One-shot simulation returning the final Bayesian posterior state.

    Parameters
    ----------
    pre_match_lambda_home / pre_match_lambda_away:
        Pre-match expected goals for each team.
    events:
        List of dicts with keys ``minute`` (int) and ``event_type`` (str).
        Optional key ``timestamp`` (float).
    final_score:
        Current or final score as (home_goals, away_goals).
    confidence:
        Prior weight — equivalent number of pre-match estimates.
    quiet_window_minutes / quiet_penalty / alert_lambda_shift:
        Passed through to BayesianLiveEngine.
    """
    # Import here to avoid a circular dependency at module load time.
    from engine.bayesian_live import BayesianLiveEngine

    prior = PreMatchPrior(
        lambda_home=pre_match_lambda_home,
        lambda_away=pre_match_lambda_away,
        confidence=confidence,
    )
    engine = BayesianLiveEngine(
        prior=prior,
        quiet_window_minutes=quiet_window_minutes,
        quiet_penalty=quiet_penalty,
        alert_lambda_shift=alert_lambda_shift,
    )
    live_events: List[LiveEvent] = [
        LiveEvent(
            minute=int(raw["minute"]),
            event_type=str(raw["event_type"]),
            timestamp=float(raw.get("timestamp", time.time())),
        )
        for raw in events
    ]
    if not live_events:
        return engine.state_at_minute(0, [], final_score)
    return engine.process_events(live_events, final_score)

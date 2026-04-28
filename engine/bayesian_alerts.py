"""
Alert generation logic for the Bayesian Live Engine.

Extracted from bayesian_live.py to keep that module under 400 LOC.
Public names are re-exported from engine.bayesian_live for backward compat.
"""

from __future__ import annotations

from typing import List

from engine.bayesian_types import BayesianAlert, BayesianState


def check_alerts(
    state: BayesianState,
    prior_lambda_home: float,
    prior_lambda_away: float,
    alert_lambda_shift: float,
    quiet_window: int,
) -> List[BayesianAlert]:
    """Compare posterior λ to pre-match λ; return alerts for significant shifts.

    Parameters
    ----------
    state:
        Current posterior BayesianState.
    prior_lambda_home / prior_lambda_away:
        Pre-match expected goals for each team.
    alert_lambda_shift:
        Fractional threshold (e.g. 0.25) that triggers an alert.
    quiet_window:
        Minutes without a shot before a QUIET_GAME alert is considered.
    """
    alerts: List[BayesianAlert] = []

    post_lh = state.posterior_lambda_home
    post_la = state.posterior_lambda_away
    shift_h = (post_lh - prior_lambda_home) / prior_lambda_home
    shift_a = (post_la - prior_lambda_away) / prior_lambda_away
    combined_shift = (
        (post_lh + post_la) - (prior_lambda_home + prior_lambda_away)
    ) / (prior_lambda_home + prior_lambda_away)
    threshold = alert_lambda_shift

    def _conf(magnitude: float) -> str:
        return "HIGH" if magnitude > 0.40 else ("MEDIUM" if magnitude > 0.30 else "LOW")

    # UNDER_VALUE: combined posterior λ well below pre-match
    if combined_shift < -threshold:
        mag = abs(combined_shift)
        alerts.append(BayesianAlert(
            state=state,
            alert_type="UNDER_VALUE",
            message=(
                f"Combined posterior λ dropped {mag:.1%} below pre-match "
                f"({prior_lambda_home + prior_lambda_away:.2f} → {post_lh + post_la:.2f}). "
                f"UNDER 2.5 looks attractive."
            ),
            edge_direction="BUY_UNDER",
            confidence=_conf(mag),
        ))

    # QUIET_GAME: same signal framed as match-tempo alert
    if combined_shift < -0.15 and state.minute >= quiet_window:
        alerts.append(BayesianAlert(
            state=state,
            alert_type="QUIET_GAME",
            message=(
                f"Quiet game pattern: λ revised down to "
                f"home={post_lh:.3f}, away={post_la:.3f}. "
                f"Draw and under markets may have value."
            ),
            edge_direction="BUY_DRAW",
            confidence="MEDIUM",
        ))

    # HOME_DRIFT: home λ dropped significantly
    if shift_h < -threshold:
        mag = abs(shift_h)
        alerts.append(BayesianAlert(
            state=state,
            alert_type="HOME_DRIFT",
            message=(
                f"Home posterior λ drifted down {mag:.1%} "
                f"({prior_lambda_home:.2f} → {post_lh:.2f}). Away advantage widening."
            ),
            edge_direction="BUY_AWAY",
            confidence=_conf(mag),
        ))

    # AWAY_SURGE: away λ increased significantly
    if shift_a > threshold:
        mag = abs(shift_a)
        alerts.append(BayesianAlert(
            state=state,
            alert_type="AWAY_SURGE",
            message=(
                f"Away posterior λ surged {mag:.1%} "
                f"({prior_lambda_away:.2f} → {post_la:.2f}). Away goal pressure rising."
            ),
            edge_direction="BUY_AWAY",
            confidence=_conf(mag),
        ))

    # HOME surge (symmetric with AWAY_SURGE)
    if shift_h > threshold:
        mag = abs(shift_h)
        alerts.append(BayesianAlert(
            state=state,
            alert_type="HOME_DRIFT",
            message=(
                f"Home posterior λ surged {mag:.1%} "
                f"({prior_lambda_home:.2f} → {post_lh:.2f}). Home dominance rising."
            ),
            edge_direction="BUY_HOME",
            confidence=_conf(mag),
        ))

    return alerts

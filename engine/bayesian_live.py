"""
bayesian_live.py — Bayesian Live Updating Engine for in-play football betting.

Starts from a pre-match Poisson model and continuously updates (λ_home, λ_away)
as in-game events arrive.  The Gamma distribution is conjugate to the Poisson
likelihood, so every goal, shot, dangerous attack and red card shifts the Gamma
posterior in closed form — no sampling required.

Key insight
-----------
"No shots in 15 minutes" is itself Bayesian evidence that λ is lower than
estimated.  Quiet-period evidence softly revises α downward, making UNDER odds
attractive live.

Mathematical model
------------------
Prior:   λ ~ Gamma(α₀, β₀)
           α₀ = pre_match_lambda × confidence
           β₀ = confidence

After observing k goals in t minutes:
           λ | data ~ Gamma(α₀ + k,  β₀ + t/90)
Posterior mean: E[λ | data] = (α₀ + k) / (β₀ + t/90)

Usage example
-------------
>>> from engine.bayesian_live import PreMatchPrior, run_bayesian_live
>>> events = [
...     {"minute": 23, "event_type": "home_goal"},
...     {"minute": 34, "event_type": "home_shot_on_target"},
...     {"minute": 55, "event_type": "away_goal"},
... ]
>>> state = run_bayesian_live(
...     pre_match_lambda_home=1.45,
...     pre_match_lambda_away=1.10,
...     events=events,
...     final_score=(1, 1),
... )
>>> print(state.summary())
"""

from __future__ import annotations

import logging
import math
import time
from typing import Dict, List, Optional, Tuple

from engine.bayesian_math import (
    compute_btts,
    compute_over_under,
    compute_win_probs,
    poisson_pmf,
)
from engine.bayesian_types import (
    _EVENT_DELTA_ALPHA,
    _MAX_GOALS,
    _MIN_ALPHA,
    BayesianAlert,
    BayesianState,
    LiveEvent,
    PreMatchPrior,
)

logger = logging.getLogger(__name__)

# Re-export for backward compatibility
__all__ = [
    "BayesianLiveEngine",
    "BayesianAlert",
    "BayesianState",
    "LiveEvent",
    "PreMatchPrior",
    "run_bayesian_live",
]


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class BayesianLiveEngine:
    """
    Bayesian live-updating engine for in-play football betting.

    Parameters
    ----------
    prior : PreMatchPrior
        Pre-match expected goals and prior confidence.
    quiet_window_minutes : int
        Number of minutes without a shot on target before triggering
        quiet-period evidence.
    quiet_penalty : float
        Maximum downward adjustment to α per quiet window.
    alert_lambda_shift : float
        Fractional shift in posterior λ that triggers an alert.
    """

    def __init__(
        self,
        prior: PreMatchPrior,
        quiet_window_minutes: int = 15,
        quiet_penalty: float = 0.08,
        alert_lambda_shift: float = 0.25,
    ) -> None:
        self.prior = prior
        self.quiet_window = quiet_window_minutes
        self.quiet_penalty = quiet_penalty
        self.alert_lambda_shift = alert_lambda_shift

        # Initialise Gamma priors from pre-match lambdas
        self._init_alpha_home: float = prior.lambda_home * prior.confidence
        self._init_beta_home: float = prior.confidence
        self._init_alpha_away: float = prior.lambda_away * prior.confidence
        self._init_beta_away: float = prior.confidence

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_event(
        self,
        event: LiveEvent,
        current_score: Tuple[int, int],
        *,
        alpha_home: Optional[float] = None,
        beta_home: Optional[float] = None,
        alpha_away: Optional[float] = None,
        beta_away: Optional[float] = None,
        events_processed: int = 0,
        last_event_minute: int = 0,
    ) -> BayesianState:
        """
        Process a single event and return the updated BayesianState.

        Callers that maintain their own state can pass explicit alpha/beta
        values; otherwise the engine starts from its initial priors.
        """
        ah = alpha_home if alpha_home is not None else self._init_alpha_home
        bh = beta_home if beta_home is not None else self._init_beta_home
        aa = alpha_away if alpha_away is not None else self._init_alpha_away
        ba = beta_away if beta_away is not None else self._init_beta_away

        delta_h, delta_a = _EVENT_DELTA_ALPHA[event.event_type]
        ah = max(_MIN_ALPHA, ah + delta_h)
        aa = max(_MIN_ALPHA, aa + delta_a)

        # Update rate parameter: β += incremental elapsed fraction since last event
        incremental_frac = (event.minute - last_event_minute) / 90.0
        bh_new = bh + incremental_frac
        ba_new = ba + incremental_frac

        return self._build_state(
            minute=event.minute,
            home_score=current_score[0],
            away_score=current_score[1],
            alpha_home=ah,
            beta_home=bh_new,
            alpha_away=aa,
            beta_away=ba_new,
            events_processed=events_processed + 1,
            last_event_minute=event.minute,
        )

    def process_events(
        self,
        events: List[LiveEvent],
        current_score: Tuple[int, int],
    ) -> BayesianState:
        """
        Process all events in chronological order, applying quiet-period
        evidence between events, and return the final BayesianState.
        """
        sorted_events = sorted(events, key=lambda ev: (ev.minute, ev.timestamp))

        ah = self._init_alpha_home
        bh = self._init_beta_home
        aa = self._init_alpha_away
        ba = self._init_beta_away

        last_shot_minute: int = 0
        events_done: int = 0
        last_event_min: int = 0

        for ev in sorted_events:
            # ---- quiet-period check before this event ----
            is_shot = ev.event_type in (
                "home_shot_on_target",
                "away_shot_on_target",
                "home_goal",
                "away_goal",
            )
            minutes_since_shot = ev.minute - last_shot_minute
            if minutes_since_shot >= self.quiet_window:
                ah, aa = self._apply_quiet_penalty(ah, aa, minutes_since_shot)

            # ---- apply event delta ----
            delta_h, delta_a = _EVENT_DELTA_ALPHA[ev.event_type]
            ah = max(_MIN_ALPHA, ah + delta_h)
            aa = max(_MIN_ALPHA, aa + delta_a)

            # Update β with incremental time elapsed since the previous event
            incremental_frac = (ev.minute - last_event_min) / 90.0
            bh += incremental_frac
            ba += incremental_frac

            if is_shot:
                last_shot_minute = ev.minute

            events_done += 1
            last_event_min = ev.minute

        # Determine the "current" minute for remaining probability calc
        current_minute = sorted_events[-1].minute if sorted_events else 0

        return self._build_state(
            minute=current_minute,
            home_score=current_score[0],
            away_score=current_score[1],
            alpha_home=ah,
            beta_home=bh,
            alpha_away=aa,
            beta_away=ba,
            events_processed=events_done,
            last_event_minute=last_event_min,
        )

    def check_alerts(self, state: BayesianState) -> List[BayesianAlert]:
        """
        Compare posterior λ to pre-match λ; emit alerts when a significant
        shift is detected.

        Returns a (possibly empty) list of BayesianAlert objects.
        """
        alerts: List[BayesianAlert] = []

        prior_lh = self.prior.lambda_home
        prior_la = self.prior.lambda_away
        post_lh = state.posterior_lambda_home
        post_la = state.posterior_lambda_away

        shift_h = (post_lh - prior_lh) / prior_lh
        shift_a = (post_la - prior_la) / prior_la
        combined_shift = ((post_lh + post_la) - (prior_lh + prior_la)) / (
            prior_lh + prior_la
        )
        threshold = self.alert_lambda_shift

        # UNDER_VALUE: both teams' posterior λ well below pre-match
        if combined_shift < -threshold:
            magnitude = abs(combined_shift)
            conf = (
                "HIGH"
                if magnitude > 0.40
                else ("MEDIUM" if magnitude > 0.30 else "LOW")
            )
            alerts.append(
                BayesianAlert(
                    state=state,
                    alert_type="UNDER_VALUE",
                    message=(
                        f"Combined posterior λ dropped {magnitude:.1%} below pre-match "
                        f"({prior_lh + prior_la:.2f} → {post_lh + post_la:.2f}). "
                        f"UNDER 2.5 looks attractive."
                    ),
                    edge_direction="BUY_UNDER",
                    confidence=conf,
                )
            )

        # QUIET_GAME: same signal but framed as match-tempo alert
        if combined_shift < -0.15 and state.minute >= self.quiet_window:
            alerts.append(
                BayesianAlert(
                    state=state,
                    alert_type="QUIET_GAME",
                    message=(
                        f"Quiet game pattern: λ revised down to "
                        f"home={post_lh:.3f}, away={post_la:.3f}. "
                        f"Draw and under markets may have value."
                    ),
                    edge_direction="BUY_DRAW",
                    confidence="MEDIUM",
                )
            )

        # HOME_DRIFT: home λ has dropped significantly
        if shift_h < -threshold:
            magnitude = abs(shift_h)
            conf = (
                "HIGH"
                if magnitude > 0.40
                else ("MEDIUM" if magnitude > 0.30 else "LOW")
            )
            alerts.append(
                BayesianAlert(
                    state=state,
                    alert_type="HOME_DRIFT",
                    message=(
                        f"Home posterior λ drifted down {magnitude:.1%} "
                        f"({prior_lh:.2f} → {post_lh:.2f}). Away advantage widening."
                    ),
                    edge_direction="BUY_AWAY",
                    confidence=conf,
                )
            )

        # AWAY_SURGE: away λ increased significantly
        if shift_a > threshold:
            magnitude = abs(shift_a)
            conf = (
                "HIGH"
                if magnitude > 0.40
                else ("MEDIUM" if magnitude > 0.30 else "LOW")
            )
            alerts.append(
                BayesianAlert(
                    state=state,
                    alert_type="AWAY_SURGE",
                    message=(
                        f"Away posterior λ surged {magnitude:.1%} "
                        f"({prior_la:.2f} → {post_la:.2f}). Away goal pressure rising."
                    ),
                    edge_direction="BUY_AWAY",
                    confidence=conf,
                )
            )

        # HOME surge (symmetric)
        if shift_h > threshold:
            magnitude = abs(shift_h)
            conf = (
                "HIGH"
                if magnitude > 0.40
                else ("MEDIUM" if magnitude > 0.30 else "LOW")
            )
            alerts.append(
                BayesianAlert(
                    state=state,
                    alert_type="HOME_DRIFT",
                    message=(
                        f"Home posterior λ surged {magnitude:.1%} "
                        f"({prior_lh:.2f} → {post_lh:.2f}). Home dominance rising."
                    ),
                    edge_direction="BUY_HOME",
                    confidence=conf,
                )
            )

        return alerts

    def state_at_minute(
        self,
        minute: int,
        events_so_far: List[LiveEvent],
        score: Tuple[int, int],
    ) -> BayesianState:
        """
        Rehydrate a BayesianState by replaying only events up to `minute`.
        """
        filtered = [ev for ev in events_so_far if ev.minute <= minute]
        if not filtered:
            # No events yet — advance β for elapsed time with no observations
            elapsed_frac = minute / 90.0
            return self._build_state(
                minute=minute,
                home_score=score[0],
                away_score=score[1],
                alpha_home=self._init_alpha_home,
                beta_home=self._init_beta_home + elapsed_frac,
                alpha_away=self._init_alpha_away,
                beta_away=self._init_beta_away + elapsed_frac,
                events_processed=0,
                last_event_minute=0,
            )
        interim = self.process_events(filtered, score)
        if interim.minute == minute:
            return interim
        # Advance β for the quiet stretch from the last event to the requested minute
        extra_frac = max(0.0, (minute - interim.last_event_minute) / 90.0)
        return self._build_state(
            minute=minute,
            home_score=score[0],
            away_score=score[1],
            alpha_home=interim.alpha_home,
            beta_home=interim.beta_home + extra_frac,
            alpha_away=interim.alpha_away,
            beta_away=interim.beta_away + extra_frac,
            events_processed=interim.events_processed,
            last_event_minute=interim.last_event_minute,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _apply_quiet_penalty(
        self,
        alpha_home: float,
        alpha_away: float,
        elapsed_quiet_minutes: int,
    ) -> Tuple[float, float]:
        """
        Soft downward revision of α for both teams when no shot has been
        recorded for `elapsed_quiet_minutes` minutes.
        """
        penalty = self.quiet_penalty * (elapsed_quiet_minutes / self.quiet_window)
        new_ah = max(_MIN_ALPHA, alpha_home - penalty)
        new_aa = max(_MIN_ALPHA, alpha_away - penalty)
        return new_ah, new_aa

    def _build_state(
        self,
        *,
        minute: int,
        home_score: int,
        away_score: int,
        alpha_home: float,
        beta_home: float,
        alpha_away: float,
        beta_away: float,
        events_processed: int,
        last_event_minute: int,
    ) -> BayesianState:
        """Construct BayesianState with all derived probabilities."""
        post_lh = alpha_home / beta_home
        post_la = alpha_away / beta_away

        remaining_frac = max(0.0, (90 - minute) / 90.0)
        rem_lh = post_lh * remaining_frac
        rem_la = post_la * remaining_frac

        p_hw, p_d, p_aw = compute_win_probs(
            rem_lh, rem_la, home_score, away_score
        )
        already_scored = home_score + away_score
        p_over, p_under = compute_over_under(
            rem_lh, rem_la, already_scored, threshold=2.5
        )
        p_btts = compute_btts(rem_lh, rem_la, home_score, away_score)

        return BayesianState(
            minute=minute,
            home_score=home_score,
            away_score=away_score,
            alpha_home=alpha_home,
            beta_home=beta_home,
            alpha_away=alpha_away,
            beta_away=beta_away,
            posterior_lambda_home=post_lh,
            posterior_lambda_away=post_la,
            p_home_win=p_hw,
            p_draw=p_d,
            p_away_win=p_aw,
            p_over_25=p_over,
            p_under_25=p_under,
            p_btts=p_btts,
            events_processed=events_processed,
            last_event_minute=last_event_minute,
        )

    # _poisson_pmf, _compute_win_probs, _compute_over_under, _compute_btts
    # extracted to engine.bayesian_math as module-level functions.


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


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
    """
    One-shot simulation returning the final Bayesian state.

    Parameters
    ----------
    pre_match_lambda_home : float
        Pre-match expected goals for the home team.
    pre_match_lambda_away : float
        Pre-match expected goals for the away team.
    events : list of dict
        Each dict must have keys ``"minute"`` (int) and ``"event_type"`` (str).
        Optional key ``"timestamp"`` (float) may also be provided.
    final_score : tuple of (int, int)
        Current or final score as (home_goals, away_goals).
    confidence : float
        Prior weight — equivalent number of pre-match estimates to anchor to.
    quiet_window_minutes : int
        Minutes without a shot before quiet-period penalty is applied.
    quiet_penalty : float
        Max α reduction per quiet window.
    alert_lambda_shift : float
        Fractional shift that triggers an alert.

    Returns
    -------
    BayesianState
        The fully updated posterior state after processing all events.

    Example
    -------
    >>> state = run_bayesian_live(
    ...     pre_match_lambda_home=1.45,
    ...     pre_match_lambda_away=1.10,
    ...     events=[
    ...         {"minute": 23, "event_type": "home_goal"},
    ...         {"minute": 55, "event_type": "away_shot_on_target"},
    ...     ],
    ...     final_score=(1, 0),
    ... )
    >>> print(state.summary())
    """
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
    live_events: List[LiveEvent] = []
    for raw in events:
        live_events.append(
            LiveEvent(
                minute=int(raw["minute"]),
                event_type=str(raw["event_type"]),
                timestamp=float(raw.get("timestamp", time.time())),
            )
        )
    if not live_events:
        # Return a prior-only state at minute 0
        return engine.state_at_minute(0, [], final_score)
    return engine.process_events(live_events, final_score)


# ---------------------------------------------------------------------------
# Module self-test (python -m engine.bayesian_live)
# ---------------------------------------------------------------------------


def _self_test() -> None:
    """Quick smoke test printed to stdout."""
    events_data = [
        {"minute": 12, "event_type": "home_dangerous_attack"},
        {"minute": 18, "event_type": "home_shot_on_target"},
        {"minute": 23, "event_type": "home_goal"},
        {"minute": 31, "event_type": "away_shot_on_target"},
        {"minute": 44, "event_type": "corner_home"},
        {"minute": 55, "event_type": "away_goal"},
        {"minute": 67, "event_type": "home_red_card"},
        {"minute": 72, "event_type": "away_dangerous_attack"},
        {"minute": 78, "event_type": "away_shot_on_target"},
    ]
    state = run_bayesian_live(
        pre_match_lambda_home=1.45,
        pre_match_lambda_away=1.10,
        events=events_data,
        final_score=(1, 1),
    )
    print(state.summary())

    prior = PreMatchPrior(lambda_home=1.45, lambda_away=1.10)
    engine = BayesianLiveEngine(prior=prior)
    live_events = [
        LiveEvent(minute=ev["minute"], event_type=ev["event_type"])
        for ev in events_data
    ]
    alerts = engine.check_alerts(state)
    if alerts:
        print("\nAlerts generated:")
        for alert in alerts:
            print(f"  {alert}")
    else:
        print("\nNo alerts generated for this scenario.")

    # Rehydrate state at minute 45
    half_time_state = engine.state_at_minute(45, live_events, (1, 0))
    print(f"\nState at half-time (45'):\n{half_time_state.summary()}")


if __name__ == "__main__":
    _self_test()

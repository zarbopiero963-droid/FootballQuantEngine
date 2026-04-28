"""
Bayesian Live Updating Engine for in-play football betting.

Starts from a pre-match Poisson model and continuously updates (λ_home, λ_away)
as in-game events arrive.  The Gamma distribution is conjugate to the Poisson
likelihood, so every goal/shot/card shifts the Gamma posterior in closed form.

Key insight: "No shots in 15 minutes" is Bayesian evidence that λ is lower
than estimated — quiet-period evidence softly revises α downward.

Mathematical model
------------------
Prior:   λ ~ Gamma(α₀, β₀)   where α₀ = lambda × confidence, β₀ = confidence
After k goals in t minutes:  λ | data ~ Gamma(α₀+k, β₀+t/90)
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

from engine.bayesian_alerts import check_alerts as _check_alerts_fn
from engine.bayesian_math import (
    compute_btts,
    compute_over_under,
    compute_win_probs,
)
from engine.bayesian_runner import run_bayesian_live
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


class BayesianLiveEngine:
    """
    Bayesian live-updating engine for in-play football betting.

    Parameters
    ----------
    prior : PreMatchPrior
        Pre-match expected goals and prior confidence.
    quiet_window_minutes : int
        Minutes without a shot on target before quiet-period evidence is applied.
    quiet_penalty : float
        Maximum downward α adjustment per quiet window.
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
        """Process a single event and return the updated BayesianState."""
        ah = alpha_home if alpha_home is not None else self._init_alpha_home
        bh = beta_home if beta_home is not None else self._init_beta_home
        aa = alpha_away if alpha_away is not None else self._init_alpha_away
        ba = beta_away if beta_away is not None else self._init_beta_away

        delta_h, delta_a = _EVENT_DELTA_ALPHA[event.event_type]
        ah = max(_MIN_ALPHA, ah + delta_h)
        aa = max(_MIN_ALPHA, aa + delta_a)

        incremental_frac = (event.minute - last_event_minute) / 90.0
        return self._build_state(
            minute=event.minute,
            home_score=current_score[0],
            away_score=current_score[1],
            alpha_home=ah,
            beta_home=bh + incremental_frac,
            alpha_away=aa,
            beta_away=ba + incremental_frac,
            events_processed=events_processed + 1,
            last_event_minute=event.minute,
        )

    def process_events(
        self,
        events: List[LiveEvent],
        current_score: Tuple[int, int],
    ) -> BayesianState:
        """Process all events in chronological order; return final BayesianState."""
        sorted_events = sorted(events, key=lambda ev: (ev.minute, ev.timestamp))

        ah = self._init_alpha_home
        bh = self._init_beta_home
        aa = self._init_alpha_away
        ba = self._init_beta_away

        last_shot_minute: int = 0
        events_done: int = 0
        last_event_min: int = 0

        for ev in sorted_events:
            is_shot = ev.event_type in (
                "home_shot_on_target",
                "away_shot_on_target",
                "home_goal",
                "away_goal",
            )
            minutes_since_shot = ev.minute - last_shot_minute
            if minutes_since_shot >= self.quiet_window:
                ah, aa = self._apply_quiet_penalty(ah, aa, minutes_since_shot)

            delta_h, delta_a = _EVENT_DELTA_ALPHA[ev.event_type]
            ah = max(_MIN_ALPHA, ah + delta_h)
            aa = max(_MIN_ALPHA, aa + delta_a)

            incremental_frac = (ev.minute - last_event_min) / 90.0
            bh += incremental_frac
            ba += incremental_frac

            if is_shot:
                last_shot_minute = ev.minute
            events_done += 1
            last_event_min = ev.minute

        current_minute = sorted_events[-1].minute if sorted_events else 0
        return self._build_state(
            minute=current_minute,
            home_score=current_score[0],
            away_score=current_score[1],
            alpha_home=ah, beta_home=bh,
            alpha_away=aa, beta_away=ba,
            events_processed=events_done,
            last_event_minute=last_event_min,
        )

    def check_alerts(self, state: BayesianState) -> List[BayesianAlert]:
        """Compare posterior λ to pre-match λ; return alerts for significant shifts."""
        return _check_alerts_fn(
            state,
            self.prior.lambda_home,
            self.prior.lambda_away,
            self.alert_lambda_shift,
            self.quiet_window,
        )

    def state_at_minute(
        self,
        minute: int,
        events_so_far: List[LiveEvent],
        score: Tuple[int, int],
    ) -> BayesianState:
        """Rehydrate a BayesianState by replaying only events up to `minute`."""
        filtered = [ev for ev in events_so_far if ev.minute <= minute]
        if not filtered:
            elapsed_frac = minute / 90.0
            return self._build_state(
                minute=minute,
                home_score=score[0], away_score=score[1],
                alpha_home=self._init_alpha_home,
                beta_home=self._init_beta_home + elapsed_frac,
                alpha_away=self._init_alpha_away,
                beta_away=self._init_beta_away + elapsed_frac,
                events_processed=0, last_event_minute=0,
            )
        interim = self.process_events(filtered, score)
        if interim.minute == minute:
            return interim
        extra_frac = max(0.0, (minute - interim.last_event_minute) / 90.0)
        return self._build_state(
            minute=minute,
            home_score=score[0], away_score=score[1],
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
        """Soft downward revision of α when no shot recorded for elapsed minutes."""
        penalty = self.quiet_penalty * (elapsed_quiet_minutes / self.quiet_window)
        return max(_MIN_ALPHA, alpha_home - penalty), max(_MIN_ALPHA, alpha_away - penalty)

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

        p_hw, p_d, p_aw = compute_win_probs(rem_lh, rem_la, home_score, away_score)
        already_scored = home_score + away_score
        p_over, p_under = compute_over_under(rem_lh, rem_la, already_scored, threshold=2.5)
        p_btts = compute_btts(rem_lh, rem_la, home_score, away_score)

        return BayesianState(
            minute=minute,
            home_score=home_score, away_score=away_score,
            alpha_home=alpha_home, beta_home=beta_home,
            alpha_away=alpha_away, beta_away=beta_away,
            posterior_lambda_home=post_lh, posterior_lambda_away=post_la,
            p_home_win=p_hw, p_draw=p_d, p_away_win=p_aw,
            p_over_25=p_over, p_under_25=p_under, p_btts=p_btts,
            events_processed=events_processed,
            last_event_minute=last_event_minute,
        )

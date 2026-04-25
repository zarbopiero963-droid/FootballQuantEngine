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
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Increment to Gamma shape (α) for each event type
_EVENT_DELTA_ALPHA: Dict[str, Tuple[float, float]] = {
    "home_goal": (1.00, 0.00),
    "away_goal": (0.00, 1.00),
    "home_shot_on_target": (0.15, 0.00),
    "away_shot_on_target": (0.00, 0.15),
    "home_dangerous_attack": (0.05, 0.00),
    "away_dangerous_attack": (0.00, 0.05),
    "home_red_card": (0.00, 0.20),  # home weakened → away benefit
    "away_red_card": (0.20, 0.00),  # away weakened → home benefit
    "corner_home": (0.03, 0.00),
    "corner_away": (0.00, 0.03),
}

_MIN_ALPHA: float = 0.5  # floor on α — prevents λ collapsing to zero
_MAX_GOALS: int = 10  # poisson score matrix dimension (captures >99.999% of mass)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class PreMatchPrior:
    """Pre-match Poisson estimates used to build Gamma priors."""

    lambda_home: float
    lambda_away: float
    confidence: float = 3.0  # equivalent prior matches

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
        """Return a compact ASCII summary of the current Bayesian state."""
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
    alert_type: str  # "UNDER_VALUE" | "HOME_DRIFT" | "AWAY_SURGE" | "QUIET_GAME"
    message: str
    edge_direction: str  # "BUY_UNDER" | "BUY_HOME" | "BUY_AWAY" | "BUY_DRAW"
    confidence: str  # "HIGH" | "MEDIUM" | "LOW"

    def __str__(self) -> str:
        return (
            f"[{self.confidence}] {self.alert_type}: {self.message} "
            f"→ {self.edge_direction}"
        )


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

        # Update rate parameter: β += elapsed_fraction
        elapsed_frac = event.minute / 90.0
        bh_new = bh + elapsed_frac
        ba_new = ba + elapsed_frac

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

            # Update β with full elapsed fraction at this event minute
            elapsed_frac = ev.minute / 90.0
            bh = bh + elapsed_frac
            ba = ba + elapsed_frac

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
            # No events yet — return prior-only state
            return self._build_state(
                minute=minute,
                home_score=score[0],
                away_score=score[1],
                alpha_home=self._init_alpha_home,
                beta_home=self._init_beta_home,
                alpha_away=self._init_alpha_away,
                beta_away=self._init_beta_away,
                events_processed=0,
                last_event_minute=0,
            )
        return self.process_events(filtered, score)

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

        p_hw, p_d, p_aw = self._compute_win_probs(
            rem_lh, rem_la, home_score, away_score
        )
        already_scored = home_score + away_score
        p_over, p_under = self._compute_over_under(
            rem_lh, rem_la, already_scored, threshold=2.5
        )
        p_btts = self._compute_btts(rem_lh, rem_la, home_score, away_score)

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

    def _poisson_pmf(self, k: int, lam: float) -> float:
        """Compute P(X = k) for X ~ Poisson(lam)."""
        if lam <= 0.0:
            return 1.0 if k == 0 else 0.0
        return math.exp(-lam) * (lam**k) / math.factorial(k)

    def _compute_win_probs(
        self,
        rem_lh: float,
        rem_la: float,
        home_score: int,
        away_score: int,
    ) -> Tuple[float, float, float]:
        """
        Compute P(home win), P(draw), P(away win) using a Poisson score
        matrix for remaining goals (0..MAX_GOALS each).

        Takes current score into account so a 1-0 home side needs 0 more
        goals to maintain a winning position, etc.
        """
        p_hw = 0.0
        p_d = 0.0
        p_aw = 0.0
        for gh in range(_MAX_GOALS + 1):
            p_gh = self._poisson_pmf(gh, rem_lh)
            for ga in range(_MAX_GOALS + 1):
                p_ga = self._poisson_pmf(ga, rem_la)
                joint = p_gh * p_ga
                total_home = home_score + gh
                total_away = away_score + ga
                if total_home > total_away:
                    p_hw += joint
                elif total_home == total_away:
                    p_d += joint
                else:
                    p_aw += joint
        return p_hw, p_d, p_aw

    def _compute_over_under(
        self,
        rem_lh: float,
        rem_la: float,
        already_scored: int,
        threshold: float,
    ) -> Tuple[float, float]:
        """
        Compute P(total goals > threshold) and P(total goals <= threshold).

        already_scored is the number of goals already in the match.
        """
        p_over = 0.0
        for gh in range(_MAX_GOALS + 1):
            p_gh = self._poisson_pmf(gh, rem_lh)
            for ga in range(_MAX_GOALS + 1):
                p_ga = self._poisson_pmf(ga, rem_la)
                total = already_scored + gh + ga
                if total > threshold:
                    p_over += p_gh * p_ga
        p_under = max(0.0, 1.0 - p_over)
        return p_over, p_under

    def _compute_btts(
        self,
        rem_lh: float,
        rem_la: float,
        home_score: int,
        away_score: int,
    ) -> float:
        """
        Compute P(both teams to score), accounting for current score.

        If a team has already scored, they only need 0 more for their
        half of the condition to be satisfied.
        """
        # P(home scores at least 1 more OR home already scored)
        if home_score > 0:
            p_home_scores = 1.0
        else:
            p_home_scores = 1.0 - self._poisson_pmf(0, rem_lh)

        # P(away scores at least 1 more OR away already scored)
        if away_score > 0:
            p_away_scores = 1.0
        else:
            p_away_scores = 1.0 - self._poisson_pmf(0, rem_la)

        return p_home_scores * p_away_scores


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

"""
correlated_parlay.py
====================
Exploits bookmakers' failure to model correlations between same-game events.

Bookmakers build Bet Builder / Same-Game Parlay prices by multiplying
independent probabilities:

    P_book(A∩B∩C) = P(A) × P(B) × P(C)      [independence assumption — wrong]

We compute the TRUE joint probability from a Poisson score matrix and an
adjusted card distribution, then compare to identify edges.
"""

from __future__ import annotations

import itertools
import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _poisson_pmf(k: int, lam: float) -> float:
    """Poisson probability mass function P(X=k) for mean *lam*."""
    if lam <= 0.0:
        return 1.0 if k == 0 else 0.0
    # Use log-space to avoid overflow for large k / lam
    return math.exp(-lam + k * math.log(lam) - math.lgamma(k + 1))


def _build_score_matrix(
    lh: float, la: float, max_goals: int
) -> Dict[Tuple[int, int], float]:
    """
    Build a joint Poisson PMF over (home_goals, away_goals) and renormalise
    so that the truncated distribution sums to 1.0.
    """
    raw: Dict[Tuple[int, int], float] = {}
    for hg in range(max_goals + 1):
        for ag in range(max_goals + 1):
            raw[(hg, ag)] = _poisson_pmf(hg, lh) * _poisson_pmf(ag, la)
    total = sum(raw.values())
    if total <= 0.0:
        raise ValueError("Score matrix has zero probability mass.")
    return {k: v / total for k, v in raw.items()}


def _inv(p: float) -> float:
    """Safe 1/p; raises if p is effectively zero."""
    if p <= 0.0:
        raise ValueError(f"Cannot invert probability p={p}")
    return 1.0 / p


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class SingleEvent:
    """One leg of a same-game parlay."""

    name: str
    """Human-readable label, e.g. 'Home Win'."""

    event_type: str
    """
    One of:
      '1x2_home' | '1x2_draw' | '1x2_away'
      'over_goals' | 'under_goals'
      'btts_yes'  | 'btts_no'
      'cards_over' | 'cards_under'
    """

    param: float
    """Line for over/under markets (e.g. 2.5 goals).  Ignored for 1X2 / BTTS."""

    market_odds: float
    """Decimal odds offered by the bookmaker."""

    model_prob: float
    """Our model's marginal probability estimate for this event."""

    bookmaker_implied: float = field(init=False)
    """1 / market_odds — bookmaker's implied probability."""

    def __post_init__(self) -> None:
        self.bookmaker_implied = (
            1.0 / self.market_odds if self.market_odds > 1.0 else 1.0
        )


@dataclass
class ParlayLeg:
    """A single leg inside a parlay."""

    event: SingleEvent
    model_prob: float
    """Model's marginal probability for this leg."""


@dataclass
class ParlayResult:
    """Result of evaluating a multi-leg same-game parlay."""

    legs: List[ParlayLeg]
    model_joint_prob: float
    """True joint probability (correlated, from score matrix)."""
    book_joint_prob: float
    """Bookmaker's assumed joint probability (independence multiplication)."""
    book_parlay_odds: float
    """Bookmaker's parlay decimal odds (1 / book_joint_prob)."""
    fair_parlay_odds: float
    """Fair odds implied by our model (1 / model_joint_prob)."""
    edge: float
    """(model_joint_prob − book_implied) in probability space."""
    edge_pct: float
    """edge / book_joint_prob × 100."""
    value_ratio: float
    """model_joint_prob / book_joint_prob."""
    is_value: bool
    """True when edge_pct exceeds the caller's threshold."""

    def __str__(self) -> str:
        leg_names = " + ".join(leg.event.name for leg in self.legs)
        return (
            f"PARLAY [{leg_names}] "
            f"true_prob={self.model_joint_prob:.4f} "
            f"fair_odds={self.fair_parlay_odds:.2f} "
            f"book_odds={self.book_parlay_odds:.2f} "
            f"edge={self.edge_pct:+.1f}% "
            f"value_ratio={self.value_ratio:.2f}x"
        )


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------


class CorrelatedParlayEngine:
    """
    Computes true joint probabilities for same-game parlays using a Poisson
    score matrix, then compares to the bookmaker's independent multiplication.

    Parameters
    ----------
    lambda_home       : expected home goals
    lambda_away       : expected away goals
    lambda_cards_home : expected home yellow cards (Poisson; default 1.8)
    lambda_cards_away : expected away yellow cards (default 1.8)
    max_goals         : score matrix truncation (default 10)
    max_cards         : card distribution truncation (default 8)
    """

    _GOAL_EVENT_TYPES = frozenset(
        {
            "1x2_home",
            "1x2_draw",
            "1x2_away",
            "over_goals",
            "under_goals",
            "btts_yes",
            "btts_no",
        }
    )
    _CARD_EVENT_TYPES = frozenset({"cards_over", "cards_under"})

    def __init__(
        self,
        lambda_home: float,
        lambda_away: float,
        lambda_cards_home: float = 1.8,
        lambda_cards_away: float = 1.8,
        max_goals: int = 10,
        max_cards: int = 8,
    ) -> None:
        if lambda_home <= 0 or lambda_away <= 0:
            raise ValueError("Goal rate lambdas must be positive.")
        if lambda_cards_home <= 0 or lambda_cards_away <= 0:
            raise ValueError("Card rate lambdas must be positive.")

        self.lambda_home = lambda_home
        self.lambda_away = lambda_away
        self.lambda_cards_home = lambda_cards_home
        self.lambda_cards_away = lambda_cards_away
        self.max_goals = max_goals
        self.max_cards = max_cards

        # Pre-build the goal score matrix once
        self._goal_matrix: Dict[Tuple[int, int], float] = self._build_goal_matrix()

        logger.debug(
            "CorrelatedParlayEngine initialised: λh=%.3f λa=%.3f λch=%.3f λca=%.3f",
            lambda_home,
            lambda_away,
            lambda_cards_home,
            lambda_cards_away,
        )

    # ------------------------------------------------------------------
    # Matrix builders
    # ------------------------------------------------------------------

    def _build_goal_matrix(self) -> Dict[Tuple[int, int], float]:
        """Joint Poisson PMF for (home_goals, away_goals), renormalised."""
        return _build_score_matrix(self.lambda_home, self.lambda_away, self.max_goals)

    def _build_card_matrix(
        self, game_state_mult: float = 1.0
    ) -> Dict[Tuple[int, int], float]:
        """
        Joint Poisson PMF for (home_cards, away_cards).

        *game_state_mult* < 1.0 when the game is lopsided (dominant team
        reduces card rate — fewer challenges, less frustration).  Both team
        lambdas are scaled by the same multiplier.
        """
        lch = max(self.lambda_cards_home * game_state_mult, 1e-9)
        lca = max(self.lambda_cards_away * game_state_mult, 1e-9)

        raw: Dict[Tuple[int, int], float] = {}
        for ch in range(self.max_cards + 1):
            for ca in range(self.max_cards + 1):
                raw[(ch, ca)] = _poisson_pmf(ch, lch) * _poisson_pmf(ca, lca)

        total = sum(raw.values())
        return {k: v / total for k, v in raw.items()}

    # ------------------------------------------------------------------
    # Marginal probabilities from the goal matrix
    # ------------------------------------------------------------------

    def p_home_win(self) -> float:
        """P(home_goals > away_goals)."""
        return sum(p for (hg, ag), p in self._goal_matrix.items() if hg > ag)

    def p_draw(self) -> float:
        """P(home_goals == away_goals)."""
        return sum(p for (hg, ag), p in self._goal_matrix.items() if hg == ag)

    def p_away_win(self) -> float:
        """P(away_goals > home_goals)."""
        return sum(p for (hg, ag), p in self._goal_matrix.items() if ag > hg)

    def p_over_goals(self, line: float) -> float:
        """P(total goals > line)."""
        return sum(p for (hg, ag), p in self._goal_matrix.items() if hg + ag > line)

    def p_under_goals(self, line: float) -> float:
        """P(total goals < line)."""
        return sum(p for (hg, ag), p in self._goal_matrix.items() if hg + ag < line)

    def p_btts_yes(self) -> float:
        """P(both teams score)."""
        return sum(p for (hg, ag), p in self._goal_matrix.items() if hg > 0 and ag > 0)

    def p_btts_no(self) -> float:
        """P(at least one team fails to score)."""
        return 1.0 - self.p_btts_yes()

    def p_cards_over(self, line: float, game_state_mult: float = 1.0) -> float:
        """P(total cards > line)."""
        card_matrix = self._build_card_matrix(game_state_mult)
        return sum(p for (ch, ca), p in card_matrix.items() if ch + ca > line)

    def p_cards_under(self, line: float, game_state_mult: float = 1.0) -> float:
        """P(total cards < line)."""
        card_matrix = self._build_card_matrix(game_state_mult)
        return sum(p for (ch, ca), p in card_matrix.items() if ch + ca < line)

    # ------------------------------------------------------------------
    # Correlation helpers
    # ------------------------------------------------------------------

    def _card_multiplier(self, goal_diff: int) -> float:
        """
        Multiplier for card rate based on goal difference (home minus away).

        Dominant scorelines → fewer cards (less contested, less frustrated).
        """
        if goal_diff >= 3:
            return 0.65
        elif goal_diff == 2:
            return 0.78
        elif goal_diff == 1:
            return 0.88
        elif goal_diff == 0:
            return 1.00
        elif goal_diff == -1:
            return 0.88
        else:  # goal_diff <= -2
            return 0.75

    def _marginal_prob_for_event(self, event: SingleEvent) -> float:
        """Return model marginal probability for a single event."""
        et = event.event_type
        if et == "1x2_home":
            return self.p_home_win()
        elif et == "1x2_draw":
            return self.p_draw()
        elif et == "1x2_away":
            return self.p_away_win()
        elif et == "over_goals":
            return self.p_over_goals(event.param)
        elif et == "under_goals":
            return self.p_under_goals(event.param)
        elif et == "btts_yes":
            return self.p_btts_yes()
        elif et == "btts_no":
            return self.p_btts_no()
        elif et == "cards_over":
            return self.p_cards_over(event.param)
        elif et == "cards_under":
            return self.p_cards_under(event.param)
        else:
            raise ValueError(f"Unknown event_type: {event.event_type!r}")

    def _score_satisfies_goal_event(self, hg: int, ag: int, event: SingleEvent) -> bool:
        """Return True if score (hg, ag) satisfies the (non-card) event."""
        et = event.event_type
        total = hg + ag
        if et == "1x2_home":
            return hg > ag
        elif et == "1x2_draw":
            return hg == ag
        elif et == "1x2_away":
            return ag > hg
        elif et == "over_goals":
            return total > event.param
        elif et == "under_goals":
            return total < event.param
        elif et == "btts_yes":
            return hg > 0 and ag > 0
        elif et == "btts_no":
            return not (hg > 0 and ag > 0)
        else:
            raise ValueError(f"Not a goal event: {event.event_type!r}")

    def _card_event_prob_given_state(
        self,
        card_events: List[SingleEvent],
        game_state_mult: float,
    ) -> float:
        """
        P(all card events are satisfied) given a specific game-state multiplier.

        Builds the card matrix once for this multiplier, then sums the
        cells that satisfy every card event.
        """
        if not card_events:
            return 1.0

        card_matrix = self._build_card_matrix(game_state_mult)
        prob = 0.0
        for (ch, ca), p_cell in card_matrix.items():
            total_cards = ch + ca
            if all(
                (e.event_type == "cards_over" and total_cards > e.param)
                or (e.event_type == "cards_under" and total_cards < e.param)
                for e in card_events
            ):
                prob += p_cell
        return prob

    # ------------------------------------------------------------------
    # Joint probability — the key method
    # ------------------------------------------------------------------

    def joint_prob(self, events: List[SingleEvent]) -> float:
        """
        Compute the TRUE joint probability for a list of same-game events.

        Algorithm
        ---------
        For every (hg, ag) cell in the goal matrix:
          1. Check whether all *goal-based* events are satisfied.
          2. If yes, derive the card-rate multiplier from the scoreline
             (dominant results → calmer game → fewer cards).
          3. Compute P(all card events satisfied | this game state).
          4. Accumulate: prob += P(hg, ag) × P(card events | game state).

        Goal events and card events are thus correlated through the shared
        score distribution, correctly capturing effects like "Home Win &
        Under 3.5 Cards" being more likely than independence implies.
        """
        goal_events = [e for e in events if e.event_type in self._GOAL_EVENT_TYPES]
        card_events = [e for e in events if e.event_type in self._CARD_EVENT_TYPES]

        # Validate event types
        for e in events:
            if e.event_type not in self._GOAL_EVENT_TYPES | self._CARD_EVENT_TYPES:
                raise ValueError(f"Unknown event_type: {e.event_type!r}")

        joint = 0.0
        for (hg, ag), p_score in self._goal_matrix.items():
            if p_score == 0.0:
                continue

            # 1. Check all goal-based events
            if not all(
                self._score_satisfies_goal_event(hg, ag, e) for e in goal_events
            ):
                continue

            # 2. Derive card rate modifier from this scoreline
            goal_diff = hg - ag
            mult = self._card_multiplier(goal_diff)

            # 3. P(all card events | game state)
            p_cards = self._card_event_prob_given_state(card_events, mult)

            # 4. Accumulate
            joint += p_score * p_cards

        return joint

    # ------------------------------------------------------------------
    # Public evaluation API
    # ------------------------------------------------------------------

    def evaluate_parlay(
        self,
        events: List[SingleEvent],
        min_edge_pct: float = 10.0,
    ) -> ParlayResult:
        """
        Evaluate a parlay: compute true joint prob, book joint prob, and edge.

        book_joint_prob  = ∏ (1 / market_odds)   [independence assumption]
        book_parlay_odds = 1 / book_joint_prob
        fair_parlay_odds = 1 / model_joint_prob
        edge             = model_joint_prob − book_joint_prob
        edge_pct         = edge / book_joint_prob × 100
        value_ratio      = model_joint_prob / book_joint_prob
        """
        if not events:
            raise ValueError("events list must not be empty.")

        # True joint probability
        model_joint = self.joint_prob(events)

        # Bookmaker's naive joint probability
        book_joint = 1.0
        for e in events:
            book_joint *= e.bookmaker_implied

        # Derive legs (use stored model_prob from SingleEvent as marginal)
        legs = [ParlayLeg(event=e, model_prob=e.model_prob) for e in events]

        book_parlay_odds = _inv(book_joint) if book_joint > 0 else float("inf")
        fair_parlay_odds = _inv(model_joint) if model_joint > 0 else float("inf")

        edge = model_joint - book_joint
        edge_pct = (edge / book_joint * 100.0) if book_joint > 0 else 0.0
        value_ratio = (model_joint / book_joint) if book_joint > 0 else 0.0

        result = ParlayResult(
            legs=legs,
            model_joint_prob=model_joint,
            book_joint_prob=book_joint,
            book_parlay_odds=book_parlay_odds,
            fair_parlay_odds=fair_parlay_odds,
            edge=edge,
            edge_pct=edge_pct,
            value_ratio=value_ratio,
            is_value=(edge_pct > min_edge_pct),
        )
        logger.info("evaluate_parlay: %s", result)
        return result

    def find_value_parlays(
        self,
        available_events: List[SingleEvent],
        max_legs: int = 4,
        min_edge_pct: float = 15.0,
        min_legs: int = 2,
    ) -> List[ParlayResult]:
        """
        Search all combinations of *available_events* (min_legs..max_legs legs)
        for parlays with positive edge.

        Returns all ParlayResult objects where is_value=True, sorted by
        edge_pct descending.
        """
        if min_legs < 2:
            raise ValueError("min_legs must be >= 2 for a parlay.")
        if max_legs < min_legs:
            raise ValueError("max_legs must be >= min_legs.")

        value_results: List[ParlayResult] = []
        n = len(available_events)

        for num_legs in range(min_legs, min(max_legs, n) + 1):
            for combo in itertools.combinations(available_events, num_legs):
                try:
                    result = self.evaluate_parlay(
                        list(combo), min_edge_pct=min_edge_pct
                    )
                except Exception as exc:
                    logger.warning(
                        "Skipping combo %s: %s",
                        [e.name for e in combo],
                        exc,
                    )
                    continue
                if result.is_value:
                    value_results.append(result)

        value_results.sort(key=lambda r: r.edge_pct, reverse=True)
        logger.info(
            "find_value_parlays: %d value parlays found (min_edge=%.1f%%)",
            len(value_results),
            min_edge_pct,
        )
        return value_results

    # ------------------------------------------------------------------
    # Correlation matrix
    # ------------------------------------------------------------------

    def correlation_matrix(self, events: List[SingleEvent]) -> List[List[float]]:
        """
        Return an N×N matrix of pairwise Pearson correlations between events.

        corr(A, B) = (P(A∩B) − P(A)·P(B)) / √(P(A)(1−P(A)) · P(B)(1−P(B)))

        Diagonal entries are 1.0.  Perfectly certain events (p=0 or p=1) yield
        a correlation of 0.0 by convention (variance is zero).
        """
        n = len(events)
        # Precompute marginal probabilities
        marginals: List[float] = []
        for e in events:
            marginals.append(self._marginal_prob_for_event(e))

        # Initialise N×N matrix
        matrix: List[List[float]] = [[0.0] * n for _ in range(n)]

        for i in range(n):
            for j in range(n):
                if i == j:
                    matrix[i][j] = 1.0
                    continue

                p_a = marginals[i]
                p_b = marginals[j]

                var_a = p_a * (1.0 - p_a)
                var_b = p_b * (1.0 - p_b)

                if var_a <= 0.0 or var_b <= 0.0:
                    matrix[i][j] = 0.0
                    continue

                # True joint probability for the pair
                p_ab = self.joint_prob([events[i], events[j]])

                cov = p_ab - p_a * p_b
                corr = cov / math.sqrt(var_a * var_b)
                # Clamp to [-1, 1] to handle floating-point drift
                matrix[i][j] = max(-1.0, min(1.0, corr))

        return matrix


# ---------------------------------------------------------------------------
# One-shot convenience function
# ---------------------------------------------------------------------------


def build_same_game_parlay(
    lambda_home: float,
    lambda_away: float,
    events: List[SingleEvent],
    lambda_cards_home: float = 1.8,
    lambda_cards_away: float = 1.8,
    min_edge_pct: float = 15.0,
) -> ParlayResult:
    """
    One-shot convenience wrapper: construct a CorrelatedParlayEngine and
    evaluate the supplied events as a single parlay.

    Parameters
    ----------
    lambda_home / lambda_away : Poisson goal rates for each team.
    events                    : List of SingleEvent legs to price.
    lambda_cards_home/away    : Expected yellow cards per team.
    min_edge_pct              : Edge threshold for is_value flag.

    Returns
    -------
    ParlayResult with fully populated fields.
    """
    engine = CorrelatedParlayEngine(
        lambda_home=lambda_home,
        lambda_away=lambda_away,
        lambda_cards_home=lambda_cards_home,
        lambda_cards_away=lambda_cards_away,
    )
    return engine.evaluate_parlay(events, min_edge_pct=min_edge_pct)

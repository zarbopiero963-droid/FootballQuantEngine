"""
Multi-factor match ranker for value betting.

Ranking algorithm
-----------------
Each candidate bet is scored on four independent dimensions:

1. Expected Value (EV)
   EV = p × (odds - 1) - (1 - p)
   Positive EV is the primary requirement.

2. Kelly Criterion (full Kelly fraction)
   f* = (p × b - q) / b
   where b = decimal_odds - 1, q = 1 - p.
   Negative Kelly → bet has negative EV (excluded).

3. Fractional Kelly stake recommendation
   Recommended stake = kelly_fraction × KELLY_SCALE × bankroll.
   Default KELLY_SCALE = 0.25 (quarter-Kelly, conservative).

4. Composite confidence score
   composite = EV × confidence × (1 + agreement) × sharpness_bonus
   where sharpness_bonus = 1 + market_edge * 2  (market confirmation bonus)

Tiers
-----
  S  : kelly ≥ 0.06  and EV ≥ 0.10  and confidence ≥ 0.70
  A  : kelly ≥ 0.03  and EV ≥ 0.05  and confidence ≥ 0.55
  B  : kelly ≥ 0.01  and EV ≥ 0.02
  C  : kelly > 0     (positive EV but marginal)
  X  : kelly ≤ 0     (no value, filtered out by default)

Sharpe-like metric
------------------
  sharpe = EV / sqrt(p × (1 - p) + 1e-9)
  Higher Sharpe → better risk-adjusted return per unit of binary variance.

Usage
-----
    from ranking.match_ranker import MatchRanker

    ranker = MatchRanker(bankroll=1000.0, kelly_scale=0.25)
    ranked = ranker.rank(pipeline_results)
    top    = ranker.top(ranked, n=5, min_tier="A")
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_KELLY_SCALE_DEFAULT = 0.25  # fraction of full Kelly (conservative)
_MIN_ODDS = 1.05  # ignore odds below this (likely error)
_MIN_PROB = 0.01
_MAX_PROB = 0.99

_TIER_THRESHOLDS = [
    # (tier, min_kelly, min_ev, min_confidence)
    ("S", 0.060, 0.10, 0.70),
    ("A", 0.030, 0.05, 0.55),
    ("B", 0.010, 0.02, 0.00),
    ("C", 0.000, 0.00, 0.00),
]


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class RankedBet:
    """Fully scored and ranked betting opportunity."""

    # Input fields
    match_id: str
    market: str
    probability: float
    odds: float
    model_edge: float = 0.0
    market_edge: float = 0.0
    confidence: float = 0.0
    agreement: float = 0.0
    decision: str = "NO_BET"
    # Computed fields
    ev: float = 0.0
    kelly: float = 0.0
    kelly_stake: float = 0.0  # recommended € stake
    composite: float = 0.0
    sharpe: float = 0.0
    tier: str = "X"
    # Pass-through extras
    extras: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        d = {
            "match_id": self.match_id,
            "market": self.market,
            "probability": round(self.probability, 4),
            "odds": round(self.odds, 3),
            "ev": round(self.ev, 5),
            "kelly": round(self.kelly, 5),
            "kelly_stake": round(self.kelly_stake, 2),
            "composite": round(self.composite, 5),
            "sharpe": round(self.sharpe, 4),
            "tier": self.tier,
            "model_edge": round(self.model_edge, 5),
            "market_edge": round(self.market_edge, 5),
            "confidence": round(self.confidence, 4),
            "agreement": round(self.agreement, 4),
            "decision": self.decision,
        }
        d.update(self.extras)
        return d


# ---------------------------------------------------------------------------
# Core scoring functions
# ---------------------------------------------------------------------------


def expected_value(probability: float, odds: float) -> float:
    """
    EV = p × (b) - (1 - p)
    where b = decimal_odds - 1 (profit per unit staked).
    """
    p = max(_MIN_PROB, min(_MAX_PROB, probability))
    b = max(0.0, odds - 1.0)
    return p * b - (1.0 - p)


def kelly_fraction(probability: float, odds: float) -> float:
    """
    Full Kelly criterion: f* = (p×b - q) / b
    Returns 0 when EV ≤ 0 (never recommend negative-EV bets).
    Capped at 1.0 (never bet entire bankroll).
    """
    p = max(_MIN_PROB, min(_MAX_PROB, probability))
    b = max(1e-9, odds - 1.0)
    q = 1.0 - p
    f = (p * b - q) / b
    return max(0.0, min(1.0, f))


def sharpe_ratio(probability: float, ev: float) -> float:
    """
    Binary-outcome Sharpe: EV / sqrt(p×q).

    Penalises high-variance extreme probabilities relative to their EV.
    """
    p = max(_MIN_PROB, min(_MAX_PROB, probability))
    q = 1.0 - p
    return ev / math.sqrt(p * q + 1e-9)


def _tier(kf: float, ev: float, conf: float) -> str:
    for tier, min_kf, min_ev, min_conf in _TIER_THRESHOLDS:
        if kf >= min_kf and ev >= min_ev and conf >= min_conf:
            return tier
    return "X"


def _composite_score(
    ev: float,
    confidence: float,
    agreement: float,
    market_edge: float,
) -> float:
    """
    Composite quality score (higher = better).

    Factors:
    - EV:          primary profitability signal
    - Confidence:  model certainty (calibrated output of ConfidenceEngine)
    - Agreement:   cross-model consensus (ModelAgreement score)
    - Market edge: confirmation from market comparison (bonus multiplier)
    """
    if ev <= 0:
        return 0.0
    agreement_bonus = 1.0 + max(0.0, agreement) * 0.5
    market_bonus = 1.0 + max(0.0, market_edge) * 2.0
    return ev * max(0.0, confidence) * agreement_bonus * market_bonus


# ---------------------------------------------------------------------------
# Main ranker
# ---------------------------------------------------------------------------


class MatchRanker:
    """
    Ranks a list of pipeline-output bet dicts by composite quality.

    Parameters
    ----------
    bankroll     : current bankroll in currency units (for stake sizing)
    kelly_scale  : fraction of full Kelly to recommend (default 0.25 = quarter-Kelly)
    min_tier     : exclude bets below this tier level ("S","A","B","C"; None = include all)
    include_no_bet: if False (default), only score BET/WATCHLIST decisions
    """

    _TIER_ORDER = {"S": 4, "A": 3, "B": 2, "C": 1, "X": 0}

    def __init__(
        self,
        bankroll: float = 1000.0,
        kelly_scale: float = _KELLY_SCALE_DEFAULT,
        min_tier: Optional[str] = None,
        include_no_bet: bool = False,
    ) -> None:
        self._bankroll = max(1.0, bankroll)
        self._kelly_scale = max(0.01, min(1.0, kelly_scale))
        self._min_tier_order = self._TIER_ORDER.get(min_tier or "C", 1)
        self._include_no_bet = include_no_bet

    # ------------------------------------------------------------------

    def score_one(self, record: dict) -> Optional[RankedBet]:
        """
        Score a single prediction record.

        Returns None if the record should be excluded (no value / bad data).
        """
        try:
            prob = float(record.get("probability", 0.0))
            odds = float(
                record.get("odds") or record.get("bookmaker_odds") or 0.0
            )
        except (TypeError, ValueError):
            return None

        if prob <= 0 or odds < _MIN_ODDS:
            return None

        decision = str(record.get("decision", "NO_BET"))
        if not self._include_no_bet and decision == "NO_BET":
            return None

        ev = expected_value(prob, odds)
        kf = kelly_fraction(prob, odds)
        sr = sharpe_ratio(prob, ev)
        comp = _composite_score(
            ev,
            float(record.get("confidence", 0.0)),
            float(record.get("agreement", 0.0)),
            float(record.get("market_edge", 0.0)),
        )
        tier = _tier(kf, ev, float(record.get("confidence", 0.0)))

        if self._TIER_ORDER.get(tier, 0) < self._min_tier_order:
            return None

        # Collect extra pass-through fields
        known = {
            "fixture_id",
            "match_id",
            "market",
            "probability",
            "odds",
            "model_edge",
            "market_edge",
            "confidence",
            "agreement",
            "decision",
        }
        extras = {k: v for k, v in record.items() if k not in known}

        return RankedBet(
            match_id=str(record.get("fixture_id") or record.get("match_id", "")),
            market=str(record.get("market", "")),
            probability=prob,
            odds=odds,
            model_edge=float(record.get("model_edge", 0.0)),
            market_edge=float(record.get("market_edge", 0.0)),
            confidence=float(record.get("confidence", 0.0)),
            agreement=float(record.get("agreement", 0.0)),
            decision=decision,
            ev=ev,
            kelly=kf,
            kelly_stake=kf * self._kelly_scale * self._bankroll,
            composite=comp,
            sharpe=sr,
            tier=tier,
            extras=extras,
        )

    def rank(self, records: list[dict]) -> list[RankedBet]:
        """
        Score and rank all records.

        Sort order (descending priority):
        1. Tier order (S > A > B > C > X)
        2. Composite score
        3. Sharpe ratio
        4. EV
        """
        scored = [self.score_one(r) for r in records]
        valid = [b for b in scored if b is not None]
        return sorted(
            valid,
            key=lambda b: (
                self._TIER_ORDER.get(b.tier, 0),
                b.composite,
                b.sharpe,
                b.ev,
            ),
            reverse=True,
        )

    def rank_as_dicts(self, records: list[dict]) -> list[dict]:
        """rank() but returns plain dicts (JSON-serialisable)."""
        return [b.as_dict() for b in self.rank(records)]

    def top(
        self,
        ranked: list[RankedBet],
        n: int = 10,
        min_tier: Optional[str] = None,
    ) -> list[RankedBet]:
        """
        Return top-N bets, optionally filtered by minimum tier.
        """
        min_order = self._TIER_ORDER.get(min_tier or "C", 1)
        filtered = [b for b in ranked if self._TIER_ORDER.get(b.tier, 0) >= min_order]
        return filtered[:n]

    def summary(self, ranked: list[RankedBet]) -> dict:
        """
        Aggregate statistics across all ranked bets.
        """
        if not ranked:
            return {"total": 0}

        tiers = {"S": 0, "A": 0, "B": 0, "C": 0, "X": 0}
        for b in ranked:
            tiers[b.tier] = tiers.get(b.tier, 0) + 1

        evs = [b.ev for b in ranked]
        kellys = [b.kelly for b in ranked]
        stakes = [b.kelly_stake for b in ranked]
        composites = [b.composite for b in ranked]

        return {
            "total": len(ranked),
            "tiers": tiers,
            "ev_mean": round(sum(evs) / len(evs), 5),
            "ev_max": round(max(evs), 5),
            "kelly_mean": round(sum(kellys) / len(kellys), 5),
            "kelly_max": round(max(kellys), 5),
            "total_stake_recommended": round(sum(stakes), 2),
            "composite_mean": round(sum(composites) / len(composites), 5),
        }

    # ------------------------------------------------------------------
    # Bankroll management helpers
    # ------------------------------------------------------------------

    def update_bankroll(self, new_bankroll: float) -> None:
        """Update bankroll for stake recalculation."""
        self._bankroll = max(1.0, new_bankroll)

    @property
    def bankroll(self) -> float:
        return self._bankroll

from __future__ import annotations

import logging
from typing import List

from quant.value.kelly_engine import KellyEngine

logger = logging.getLogger(__name__)


class StakePolicy:

    def __init__(self):
        self.kelly = KellyEngine()

    def suggest(
        self,
        bankroll: float,
        probability: float,
        odds: float,
        confidence: float,
        fraction: float = 0.25,
    ) -> dict:
        confidence = float(confidence)

        if confidence >= 0.80:
            max_cap = 0.05
        elif confidence >= 0.65:
            max_cap = 0.035
        else:
            max_cap = 0.02

        stake = self.kelly.suggested_stake(
            bankroll=bankroll,
            probability=probability,
            odds=odds,
            fraction=fraction,
            max_bankroll_pct=max_cap,
        )

        return {
            "stake": round(stake, 2),
            "max_cap_pct": max_cap,
        }

    def suggest_portfolio(
        self,
        bets: List[dict],
        bankroll: float,
        max_bankroll_fraction: float = 0.20,
        max_stake_fraction: float = 0.05,
    ) -> dict:
        """
        Size a slate of potentially correlated bets via Markowitz optimisation.

        Each bet dict must contain:
          - ``bet_id``           : unique identifier string
          - ``model_prob``       : float in (0, 1)
          - ``odds``             : decimal odds ≥ 1.0
          - ``correlation_group``: string — bets in the same group get a higher
                                   intra-group correlation (default 0.5 same-fixture,
                                   0.3 same-competition-round, 0.1 cross-group)

        Returns a dict with:
          - ``weights``     : {bet_id: fraction_of_bankroll}
          - ``stakes``      : {bet_id: £ amount}
          - ``expected_return``  : portfolio μ
          - ``portfolio_variance``: portfolio σ²
        """
        from engine.markowitz_optimizer import MarkowitzOptimizer, BetProposal

        if not bets:
            return {"weights": {}, "stakes": {}, "expected_return": 0.0, "portfolio_variance": 0.0}

        proposals = []
        for b in bets:
            try:
                proposals.append(BetProposal(
                    bet_id=str(b["bet_id"]),
                    description=str(b.get("description", b["bet_id"])),
                    odds=float(b["odds"]),
                    model_prob=float(b["model_prob"]),
                    correlation_group=str(b.get("correlation_group", "default")),
                ))
            except (KeyError, ValueError, TypeError) as exc:
                logger.warning("suggest_portfolio: skipping malformed bet %s — %s", b, exc)

        if not proposals:
            return {"weights": {}, "stakes": {}, "expected_return": 0.0, "portfolio_variance": 0.0}

        for p in proposals:
            p.max_stake_fraction = max_stake_fraction

        optimizer = MarkowitzOptimizer(bankroll_fraction=max_bankroll_fraction)
        allocation = optimizer.optimise(proposals)

        weights_map = {b.bet_id: w for b, w in zip(allocation.bets, allocation.weights)}
        stakes = {bid: round(w * bankroll, 2) for bid, w in weights_map.items()}
        return {
            "weights": weights_map,
            "stakes": stakes,
            "expected_return": allocation.expected_return,
            "portfolio_variance": allocation.portfolio_variance,
        }

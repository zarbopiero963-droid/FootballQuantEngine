"""
Match ranking with Kelly criterion, composite score, and Sharpe-like metric.

This module provides the analysis-layer MatchRanking class (used by
engine/match_ranker.py and tests). For the full post-processing ranker
with bankroll management and tier classification, see ranking/match_ranker.py.

Metrics computed
----------------
edge        : probability - (1 / decimal_odds)   — simple market edge
ev          : p × (odds - 1) - (1 - p)           — expected value per unit
kelly       : (p × b - q) / b                     — full Kelly fraction
sharpe      : ev / sqrt(p × q)                    — risk-adjusted EV
composite   : ev × confidence × (1 + agreement)  — quality composite score
tier        : S / A / B / C / X classification
"""

from __future__ import annotations

import math

import pandas as pd

# ---------------------------------------------------------------------------
# Pure functions (used both by the class and by ranking/match_ranker.py)
# ---------------------------------------------------------------------------


def _ev(prob: float, odds: float) -> float:
    p = max(1e-6, min(1 - 1e-6, prob))
    b = max(0.0, odds - 1.0)
    return p * b - (1.0 - p)


def _kelly(prob: float, odds: float) -> float:
    p = max(1e-6, min(1 - 1e-6, prob))
    b = max(1e-9, odds - 1.0)
    q = 1.0 - p
    return max(0.0, min(1.0, (p * b - q) / b))


def _sharpe(prob: float, ev: float) -> float:
    p = max(1e-6, min(1 - 1e-6, prob))
    return ev / math.sqrt(p * (1.0 - p) + 1e-9)


def _composite(ev: float, confidence: float, agreement: float) -> float:
    if ev <= 0:
        return 0.0
    return ev * max(0.0, confidence) * (1.0 + max(0.0, agreement))


def _tier(kelly: float, ev: float, confidence: float) -> str:
    thresholds = [
        ("S", 0.060, 0.10, 0.70),
        ("A", 0.030, 0.05, 0.55),
        ("B", 0.010, 0.02, 0.00),
        ("C", 0.000, 0.00, 0.00),
    ]
    for name, min_k, min_ev, min_conf in thresholds:
        if kelly >= min_k and ev >= min_ev and confidence >= min_conf:
            return name
    return "X"


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class MatchRanking:
    """
    Ranks a list of value-bet dicts by composite quality metrics.

    The returned DataFrame includes all original columns plus:
      edge, ev, kelly, sharpe, composite, tier

    Sort order (descending priority):
      tier → composite → sharpe → ev
    """

    _TIER_ORDER = {"S": 4, "A": 3, "B": 2, "C": 1, "X": 0}

    def rank(self, value_bets: list[dict]) -> pd.DataFrame:
        """
        Rank value bets by Kelly criterion and composite score.

        Parameters
        ----------
        value_bets : list of dicts, each with at least:
                     probability (float), odds (float)
                     Optional: confidence, agreement, market_edge, decision

        Returns
        -------
        pd.DataFrame sorted by composite quality (best first).
        """
        if not value_bets:
            return pd.DataFrame()

        df = pd.DataFrame(value_bets)

        if "probability" not in df.columns or "odds" not in df.columns:
            return df

        prob = df["probability"].clip(1e-6, 1 - 1e-6).astype(float)
        odds = df["odds"].clip(lower=1.001).astype(float)
        conf = (
            df["confidence"].astype(float)
            if "confidence" in df.columns
            else pd.Series(0.0, index=df.index)
        )
        agr = (
            df["agreement"].astype(float)
            if "agreement" in df.columns
            else pd.Series(0.0, index=df.index)
        )

        df["edge"] = prob - (1.0 / odds)
        df["ev"] = prob * (odds - 1.0) - (1.0 - prob)
        df["kelly"] = ((prob * (odds - 1.0) - (1.0 - prob)) / (odds - 1.0)).clip(
            lower=0.0, upper=1.0
        )
        df["sharpe"] = df["ev"] / (prob * (1.0 - prob) + 1e-9).pow(0.5)
        df["composite"] = df["ev"].clip(lower=0) * conf * (1.0 + agr)
        df["tier"] = df.apply(
            lambda r: _tier(r["kelly"], r["ev"], r.get("confidence", 0.0)),
            axis=1,
        )
        df["tier_order"] = df["tier"].map(self._TIER_ORDER).fillna(0).astype(int)

        df = df.sort_values(
            by=["tier_order", "composite", "sharpe", "ev"],
            ascending=False,
        ).drop(columns=["tier_order"])

        return df.reset_index(drop=True)

    def top_bets(
        self,
        value_bets: list[dict],
        n: int = 10,
        min_tier: str = "B",
    ) -> pd.DataFrame:
        """
        Rank and return the top-N bets filtered by minimum tier.

        Parameters
        ----------
        n        : maximum number of rows to return
        min_tier : exclude bets below this tier ("S","A","B","C")
        """
        ranked = self.rank(value_bets)
        min_order = self._TIER_ORDER.get(min_tier, 0)
        if ranked.empty or "tier" not in ranked.columns:
            return ranked.head(n)
        filtered = ranked[ranked["tier"].map(self._TIER_ORDER).fillna(0) >= min_order]
        return filtered.head(n)

    def summary(self, value_bets: list[dict]) -> dict:
        """Return aggregate statistics across all ranked bets."""
        df = self.rank(value_bets)
        if df.empty:
            return {"total": 0}

        tier_counts = (
            df["tier"].value_counts().to_dict() if "tier" in df.columns else {}
        )
        return {
            "total": len(df),
            "tiers": tier_counts,
            "ev_mean": round(float(df["ev"].mean()), 5) if "ev" in df.columns else 0.0,
            "ev_max": round(float(df["ev"].max()), 5) if "ev" in df.columns else 0.0,
            "kelly_mean": (
                round(float(df["kelly"].mean()), 5) if "kelly" in df.columns else 0.0
            ),
            "sharpe_mean": (
                round(float(df["sharpe"].mean()), 4) if "sharpe" in df.columns else 0.0
            ),
        }

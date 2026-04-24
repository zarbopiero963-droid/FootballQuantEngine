"""
Threshold optimizer — learns optimal bet-placement thresholds from backtest history.

Problem
-------
Manual thresholds (min_edge=0.05, min_confidence=0.60, min_kelly=0.02) may
be sub-optimal.  This module runs a grid search over the threshold space on
historical backtest data and selects the combination that maximises a
composite objective function:

  score = ROI × (1 + hit_rate) × (1 - |max_drawdown|)

Subject to:
  - total_bets >= min_sample  (at least 50 bets for statistical significance)
  - hit_rate >= 0.30          (sanity guard)

Grid
----
  min_edge:        [0.02, 0.04, 0.05, 0.06, 0.08, 0.10]
  min_confidence:  [0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
  min_kelly:       [0.01, 0.02, 0.03, 0.04, 0.05]

Usage
-----
    from engine.threshold_optimizer import ThresholdOptimizer

    opt = ThresholdOptimizer()
    result = opt.optimise(backtest_df)
    print(result.best_thresholds)   # {"min_edge": 0.06, "min_confidence": 0.60, "min_kelly": 0.02}
    print(result.best_roi)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from itertools import product
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Grid definition
# ---------------------------------------------------------------------------

_GRID_EDGE = [0.02, 0.04, 0.05, 0.06, 0.08, 0.10]
_GRID_CONF = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
_GRID_KELLY = [0.01, 0.02, 0.03, 0.04, 0.05]

_MIN_SAMPLE = 50
_MIN_HIT_RATE = 0.30


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class OptimisationResult:
    best_thresholds: Dict[str, float] = field(
        default_factory=lambda: {
            "min_edge": 0.05,
            "min_confidence": 0.60,
            "min_kelly": 0.02,
        }
    )
    best_roi: float = 0.0
    best_score: float = 0.0
    best_bets: int = 0
    best_hit_rate: float = 0.0
    best_max_drawdown: float = 0.0
    n_combinations_tried: int = 0
    n_valid_combinations: int = 0
    all_results: List[Dict[str, Any]] = field(default_factory=list)

    def summary(self) -> str:
        t = self.best_thresholds
        return (
            f"ThresholdOptimizer: best edge={t['min_edge']:.2f} "
            f"conf={t['min_confidence']:.2f} kelly={t['min_kelly']:.2f} "
            f"→ ROI={self.best_roi:.2%} bets={self.best_bets} "
            f"hit={self.best_hit_rate:.2%} drawdown={self.best_max_drawdown:.2%}"
        )


# ---------------------------------------------------------------------------
# ThresholdOptimizer
# ---------------------------------------------------------------------------


class ThresholdOptimizer:
    """
    Grid-search threshold optimizer.

    Parameters
    ----------
    grid_edge       : list of min_edge values to try
    grid_confidence : list of min_confidence values to try
    grid_kelly      : list of min_kelly values to try
    min_sample      : minimum bets required for a threshold combo to be valid
    """

    def __init__(
        self,
        grid_edge: Optional[List[float]] = None,
        grid_confidence: Optional[List[float]] = None,
        grid_kelly: Optional[List[float]] = None,
        min_sample: int = _MIN_SAMPLE,
    ) -> None:
        self._grid_edge = grid_edge or _GRID_EDGE
        self._grid_conf = grid_confidence or _GRID_CONF
        self._grid_kelly = grid_kelly or _GRID_KELLY
        self._min_sample = min_sample

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def optimise(self, backtest_df: pd.DataFrame) -> OptimisationResult:
        """
        Run the grid search on historical backtest data.

        Parameters
        ----------
        backtest_df : DataFrame returned by BacktestEngine.run().
            Expected columns: home_prob, draw_prob, away_prob,
            home_odds, draw_odds, away_odds,
            actual_home_win, actual_draw, actual_away_win.
            Optional columns used when present: edge, confidence, kelly_fraction.

        Returns
        -------
        OptimisationResult with best thresholds and summary statistics.
        """
        result = OptimisationResult()

        if backtest_df is None or backtest_df.empty:
            logger.warning("ThresholdOptimizer: empty backtest_df — returning defaults")
            return result

        df = _enrich(backtest_df)

        combos = list(product(self._grid_edge, self._grid_conf, self._grid_kelly))
        result.n_combinations_tried = len(combos)

        all_rows: List[Dict[str, Any]] = []
        best_score = -1e9

        for min_edge, min_conf, min_kelly in combos:
            subset = df[
                (df["edge"] >= min_edge)
                & (df["confidence"] >= min_conf)
                & (df["kelly_fraction"] >= min_kelly)
            ]

            n = len(subset)
            if n < self._min_sample:
                continue

            hit_rate = float(subset["won"].mean())
            if hit_rate < _MIN_HIT_RATE:
                continue

            total_profit = float(subset["profit"].sum())
            total_staked = float(n)
            roi = total_profit / total_staked if total_staked > 0 else 0.0

            max_dd = _max_drawdown(subset["profit"].tolist())

            score = roi * (1.0 + hit_rate) * (1.0 - abs(max_dd))

            row = {
                "min_edge": min_edge,
                "min_confidence": min_conf,
                "min_kelly": min_kelly,
                "bets": n,
                "roi": roi,
                "hit_rate": hit_rate,
                "max_drawdown": max_dd,
                "score": score,
            }
            all_rows.append(row)
            result.n_valid_combinations += 1

            if score > best_score:
                best_score = score
                result.best_thresholds = {
                    "min_edge": min_edge,
                    "min_confidence": min_conf,
                    "min_kelly": min_kelly,
                }
                result.best_roi = roi
                result.best_score = score
                result.best_bets = n
                result.best_hit_rate = hit_rate
                result.best_max_drawdown = max_dd

        result.all_results = sorted(all_rows, key=lambda r: r["score"], reverse=True)

        logger.info(
            "ThresholdOptimizer: tried %d combos, %d valid — %s",
            result.n_combinations_tried,
            result.n_valid_combinations,
            result.summary()
            if result.n_valid_combinations > 0
            else "no valid combo found",
        )
        return result

    def top_n(self, backtest_df: pd.DataFrame, n: int = 10) -> List[Dict[str, Any]]:
        """Return the top-N threshold combinations ranked by score."""
        return self.optimise(backtest_df).all_results[:n]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _enrich(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add synthetic edge / confidence / kelly_fraction / won / profit columns
    when they are not already present, derived from model probs and market odds.
    """
    df = df.copy()

    # Determine which market each row bets on (highest model probability)
    if "selected_market" not in df.columns:
        df["selected_market"] = df[["home_prob", "draw_prob", "away_prob"]].idxmax(
            axis=1
        )
        df["selected_market"] = df["selected_market"].str.replace("_prob", "")

    # Market odds for the selected outcome
    if "selected_odds" not in df.columns:
        df["selected_odds"] = df.apply(_resolve_odds, axis=1)

    # Model probability for the selected outcome
    df["selected_prob"] = df.apply(_resolve_prob, axis=1)

    # Edge = model prob - implied prob from odds
    if "edge" not in df.columns:
        df["implied_prob"] = 1.0 / df["selected_odds"].clip(lower=1.01)
        df["edge"] = df["selected_prob"] - df["implied_prob"]

    # Confidence: use existing column or approximate from prob + edge
    if "confidence" not in df.columns:
        df["confidence"] = (
            df["selected_prob"] * 0.6 + df["edge"].clip(0) * 2.0 * 0.4
        ).clip(0, 1)

    # Kelly fraction: f* = (p*b - q) / b  where b = odds - 1
    if "kelly_fraction" not in df.columns:
        b = (df["selected_odds"] - 1.0).clip(lower=0.01)
        p = df["selected_prob"]
        q = 1.0 - p
        df["kelly_fraction"] = ((p * b - q) / b).clip(lower=0.0)

    # Won flag
    if "won" not in df.columns:
        df["won"] = _compute_won(df)

    # Profit (flat stake = 1 unit)
    if "profit" not in df.columns:
        df["profit"] = df.apply(
            lambda row: (row["selected_odds"] - 1.0) if row["won"] else -1.0, axis=1
        )

    return df


def _resolve_odds(row: pd.Series) -> float:
    m = str(row.get("selected_market", "home"))
    if m == "home":
        return float(row.get("home_odds", 2.0) or 2.0)
    if m == "draw":
        return float(row.get("draw_odds", 3.0) or 3.0)
    return float(row.get("away_odds", 2.5) or 2.5)


def _resolve_prob(row: pd.Series) -> float:
    m = str(row.get("selected_market", "home"))
    if m == "home":
        return float(row.get("home_prob", 1 / 3) or 1 / 3)
    if m == "draw":
        return float(row.get("draw_prob", 1 / 3) or 1 / 3)
    return float(row.get("away_prob", 1 / 3) or 1 / 3)


def _compute_won(df: pd.DataFrame) -> pd.Series:
    won = pd.Series(0, index=df.index)
    won[(df["selected_market"] == "home") & (df.get("actual_home_win", 0) == 1)] = 1
    won[(df["selected_market"] == "draw") & (df.get("actual_draw", 0) == 1)] = 1
    won[(df["selected_market"] == "away") & (df.get("actual_away_win", 0) == 1)] = 1
    return won


def _max_drawdown(profits: List[float]) -> float:
    """Peak-to-trough max drawdown from a list of per-bet profits."""
    if not profits:
        return 0.0
    bankroll = 100.0
    peak = bankroll
    max_dd = 0.0
    for p in profits:
        bankroll += p
        peak = max(peak, bankroll)
        dd = (bankroll - peak) / peak if peak > 0 else 0.0
        max_dd = min(max_dd, dd)
    return max_dd

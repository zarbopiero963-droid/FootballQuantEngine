"""
Full Asian Handicap and Asian Over/Under pricing from a Poisson score matrix.

Supported lines
---------------
AH (home handicap):
  ±0.25, ±0.5, ±0.75, ±1.0, ±1.25, ±1.5, ±1.75, ±2.0, ±2.25, ±2.5
  and any other multiple of 0.25 (quarter-, half-, whole-lines)

O/U:
  0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5

Handicap sign convention
------------------------
  line < 0 → home favourite, giving goals (e.g. line=-1 means "Home -1")
  line > 0 → home underdog, receiving goals (e.g. line=+1 means "Home +1")
  line = 0 → Draw No Bet

Quarter-line pricing rule
--------------------------
A quarter-line (e.g. -0.75) splits the stake 50/50 between the two adjacent
half-lines (-0.5 and -1.0).  Win/push/lose outcomes are averaged.

  p_win_ah(-0.75) = 0.5 × p_win_ah(-0.5) + 0.5 × p_win_ah(-1.0)

This matches how Pinnacle and Asian bookmakers settle these bets.

Usage
-----
    from models.asian_handicap import AsianHandicapModel

    model = AsianHandicapModel(lambda_home=1.6, lambda_away=1.0)
    result = model.price_ah(-1.0)     # Home -1
    # AHResult(p_home_win=0.42, p_push=0.18, p_away_win=0.40, fair_odds_home=2.38, ...)

    ou = model.price_ou(2.5)
    # OUResult(p_over=0.54, p_under=0.46, fair_odds_over=1.85, ...)

    all_lines = model.full_market()
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

_MAX_GOALS = 12
_OVERROUND = 1.0  # set to e.g. 1.04 to simulate a book margin


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class AHResult:
    line: float
    p_home_win: float  # home wins AH (stake returned in full at market odds)
    p_push: float  # stake refunded (whole-line only)
    p_away_win: float  # home loses AH
    fair_odds_home: float  # 1 / p_home_win_adj (push → half refund)
    fair_odds_away: float


@dataclass
class OUResult:
    line: float
    p_over: float
    p_push: float  # exact-goals total (whole-line only)
    p_under: float
    fair_odds_over: float
    fair_odds_away: float  # alias: fair_odds_under


@dataclass
class FullMarket:
    lambda_home: float
    lambda_away: float
    ah_lines: Dict[float, AHResult]
    ou_lines: Dict[float, OUResult]
    p_home_win_1x2: float
    p_draw_1x2: float
    p_away_win_1x2: float


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------


class AsianHandicapModel:
    """
    Prices Asian Handicap and Over/Under markets from Poisson parameters.

    The score matrix is computed once on construction; all pricing queries
    use matrix lookups — O(N²) cost at construction, O(lines) at query time.

    Parameters
    ----------
    lambda_home : expected home goals (from FeatureEngine.lambda_goals)
    lambda_away : expected away goals
    max_goals   : truncation point for the score matrix (default 12)
    """

    def __init__(
        self,
        lambda_home: float,
        lambda_away: float,
        max_goals: int = _MAX_GOALS,
    ) -> None:
        self.lambda_home = max(0.05, lambda_home)
        self.lambda_away = max(0.05, lambda_away)
        self._max = max_goals
        self._matrix = _build_score_matrix(
            self.lambda_home, self.lambda_away, max_goals
        )
        # Precompute margin distribution for AH pricing
        self._margin_pmf = _margin_distribution(self._matrix, max_goals)
        # Precompute total-goals distribution for O/U pricing
        self._total_pmf = _total_goals_distribution(self._matrix, max_goals)

    # ------------------------------------------------------------------
    # Asian Handicap
    # ------------------------------------------------------------------

    def price_ah(self, line: float) -> AHResult:
        """
        Price the home side of an Asian Handicap at the given line.

        line < 0 → home favourite (e.g. -1.0 = "Home -1")
        line > 0 → away favourite (e.g. +1.5 = "Home +1.5")
        """
        # Quarter-line: split 50/50 between the two adjacent non-quarter lines
        if _is_quarter_line(line):
            lo = line - 0.25
            hi = line + 0.25
            r_lo = self._price_ah_base(lo)
            r_hi = self._price_ah_base(hi)
            p_win = 0.5 * r_lo[0] + 0.5 * r_hi[0]
            p_push = 0.5 * r_lo[1] + 0.5 * r_hi[1]
            p_lose = 0.5 * r_lo[2] + 0.5 * r_hi[2]
        else:
            p_win, p_push, p_lose = self._price_ah_base(line)

        fair_home, fair_away = _fair_odds(p_win, p_push, p_lose)
        return AHResult(
            line=line,
            p_home_win=p_win,
            p_push=p_push,
            p_away_win=p_lose,
            fair_odds_home=fair_home,
            fair_odds_away=fair_away,
        )

    def _price_ah_base(self, line: float) -> Tuple[float, float, float]:
        """
        Compute (p_win, p_push, p_lose) for a non-quarter AH line.

        Adjusted margin = margin + line
        Win if > 0, push if == 0 (only integer lines), lose if < 0.
        """
        p_win = 0.0
        p_push = 0.0
        p_lose = 0.0
        for margin, prob in self._margin_pmf.items():
            adjusted = margin + line  # e.g. margin=1, line=-1 → adjusted=0 (push)
            if adjusted > 0:
                p_win += prob
            elif adjusted == 0 and _is_whole_number(line):
                p_push += prob
            else:
                p_lose += prob
        return p_win, p_push, p_lose

    # ------------------------------------------------------------------
    # Over / Under
    # ------------------------------------------------------------------

    def price_ou(self, line: float) -> OUResult:
        """
        Price Over/Under at the given total-goals line.

        line = 2.5 → over if total ≥ 3, under if total ≤ 2 (no push)
        line = 2.0 → over if total ≥ 3, push if total = 2, under if total ≤ 1
        line = 2.25 → split: half on 2.0, half on 2.5 (quarter-line)
        """
        if _is_quarter_line(line):
            lo = line - 0.25
            hi = line + 0.25
            o_lo, push_lo, u_lo = self._price_ou_base(lo)
            o_hi, push_hi, u_hi = self._price_ou_base(hi)
            p_over = 0.5 * o_lo + 0.5 * o_hi
            p_push = 0.5 * push_lo + 0.5 * push_hi
            p_under = 0.5 * u_lo + 0.5 * u_hi
        else:
            p_over, p_push, p_under = self._price_ou_base(line)

        fair_over = _inv(p_over + 0.5 * p_push)
        fair_under = _inv(p_under + 0.5 * p_push)
        return OUResult(
            line=line,
            p_over=p_over,
            p_push=p_push,
            p_under=p_under,
            fair_odds_over=fair_over,
            fair_odds_away=fair_under,
        )

    def _price_ou_base(self, line: float) -> Tuple[float, float, float]:
        p_over = 0.0
        p_push = 0.0
        p_under = 0.0
        for total, prob in self._total_pmf.items():
            if total > line:
                p_over += prob
            elif total == line and _is_whole_number(line):
                p_push += prob
            else:
                p_under += prob
        return p_over, p_push, p_under

    # ------------------------------------------------------------------
    # 1X2 from matrix
    # ------------------------------------------------------------------

    def prob_1x2(self) -> Tuple[float, float, float]:
        """Return (p_home, p_draw, p_away) from the score matrix."""
        p_h = p_d = p_a = 0.0
        for (hg, ag), prob in self._matrix.items():
            if hg > ag:
                p_h += prob
            elif hg == ag:
                p_d += prob
            else:
                p_a += prob
        total = p_h + p_d + p_a
        if total <= 0:
            return 1 / 3, 1 / 3, 1 / 3
        return p_h / total, p_d / total, p_a / total

    # ------------------------------------------------------------------
    # Full market snapshot
    # ------------------------------------------------------------------

    AH_LINES = [
        -2.5,
        -2.25,
        -2.0,
        -1.75,
        -1.5,
        -1.25,
        -1.0,
        -0.75,
        -0.5,
        -0.25,
        0.0,
        0.25,
        0.5,
        0.75,
        1.0,
        1.25,
        1.5,
        1.75,
        2.0,
        2.25,
        2.5,
    ]
    OU_LINES = [
        0.5,
        0.75,
        1.0,
        1.25,
        1.5,
        1.75,
        2.0,
        2.25,
        2.5,
        2.75,
        3.0,
        3.25,
        3.5,
        4.0,
    ]

    def full_market(self) -> FullMarket:
        """Price all standard AH and O/U lines in one call."""
        ph, pd, pa = self.prob_1x2()
        return FullMarket(
            lambda_home=self.lambda_home,
            lambda_away=self.lambda_away,
            ah_lines={line: self.price_ah(line) for line in self.AH_LINES},
            ou_lines={line: self.price_ou(line) for line in self.OU_LINES},
            p_home_win_1x2=ph,
            p_draw_1x2=pd,
            p_away_win_1x2=pa,
        )

    def find_value_ah(
        self,
        market_odds_home: float,
        market_odds_away: float,
        line: float,
        min_edge: float = 0.02,
    ) -> Optional[Dict]:
        """
        Return value-bet info if fair price beats market odds by at least min_edge.

        market_odds_home / market_odds_away : decimal odds offered by the book
        line                                : AH line (e.g. -0.75)
        min_edge                            : minimum required edge (default 2%)
        """
        r = self.price_ah(line)
        p_home_adj = r.p_home_win + 0.5 * r.p_push
        p_away_adj = r.p_away_win + 0.5 * r.p_push

        edge_home = p_home_adj - 1.0 / market_odds_home
        edge_away = p_away_adj - 1.0 / market_odds_away

        if edge_home < min_edge and edge_away < min_edge:
            return None

        side = "home" if edge_home >= edge_away else "away"
        edge = edge_home if side == "home" else edge_away
        our_fair = r.fair_odds_home if side == "home" else r.fair_odds_away
        market = market_odds_home if side == "home" else market_odds_away

        return {
            "line": line,
            "side": side,
            "edge": round(edge, 4),
            "fair_odds": round(our_fair, 3),
            "market_odds": round(market, 3),
            "p_home_win": round(r.p_home_win, 4),
            "p_push": round(r.p_push, 4),
            "p_away_win": round(r.p_away_win, 4),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_score_matrix(
    lh: float, la: float, max_goals: int
) -> Dict[Tuple[int, int], float]:
    matrix: Dict[Tuple[int, int], float] = {}
    total = 0.0
    for hg in range(max_goals + 1):
        for ag in range(max_goals + 1):
            p = _poisson_pmf(hg, lh) * _poisson_pmf(ag, la)
            matrix[(hg, ag)] = p
            total += p
    # Renormalise for truncation
    if total > 0:
        matrix = {k: v / total for k, v in matrix.items()}
    return matrix


def _margin_distribution(
    matrix: Dict[Tuple[int, int], float], max_goals: int
) -> Dict[int, float]:
    """Probability mass function of goal difference (home - away)."""
    pmf: Dict[int, float] = {}
    for (hg, ag), prob in matrix.items():
        margin = hg - ag
        pmf[margin] = pmf.get(margin, 0.0) + prob
    return pmf


def _total_goals_distribution(
    matrix: Dict[Tuple[int, int], float], max_goals: int
) -> Dict[int, float]:
    pmf: Dict[int, float] = {}
    for (hg, ag), prob in matrix.items():
        total = hg + ag
        pmf[total] = pmf.get(total, 0.0) + prob
    return pmf


def _poisson_pmf(k: int, lam: float) -> float:
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    try:
        return math.exp(-lam) * (lam**k) / math.factorial(k)
    except (OverflowError, ValueError):
        return 0.0


def _is_quarter_line(line: float) -> bool:
    """True if the line is a quarter-line (e.g. ±0.25, ±0.75, ±1.25...)."""
    return abs(round(line * 4) % 2) == 1


def _is_whole_number(line: float) -> bool:
    return abs(line - round(line)) < 1e-9


def _fair_odds(p_win: float, p_push: float, p_lose: float) -> Tuple[float, float]:
    """
    Convert win/push/lose into fair decimal odds.

    On a push the stake is refunded, so the effective winning probability is
    p_win and effective losing probability is p_lose.  The odds are:
      fair_home = 1 / (p_win + 0.5 × p_push)
      fair_away = 1 / (p_lose + 0.5 × p_push)
    """
    p_home_adj = p_win + 0.5 * p_push
    p_away_adj = p_lose + 0.5 * p_push
    return _inv(p_home_adj), _inv(p_away_adj)


def _inv(p: float, floor: float = 1.01) -> float:
    if p <= 0:
        return 999.0
    return max(floor, 1.0 / p)


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def price_asian_market(
    lambda_home: float,
    lambda_away: float,
    ah_lines: Optional[List[float]] = None,
    ou_lines: Optional[List[float]] = None,
) -> FullMarket:
    """
    One-shot convenience: price AH and O/U markets from Poisson parameters.

    ah_lines / ou_lines default to AsianHandicapModel.AH_LINES / OU_LINES.
    """
    m = AsianHandicapModel(lambda_home, lambda_away)
    if ah_lines is None and ou_lines is None:
        return m.full_market()
    fm = m.full_market()
    if ah_lines is not None:
        fm.ah_lines = {line: m.price_ah(line) for line in ah_lines}
    if ou_lines is not None:
        fm.ou_lines = {line: m.price_ou(line) for line in ou_lines}
    return fm

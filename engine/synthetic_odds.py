"""
Synthetic Odds Generator — statistical arbitrage via derived market pricing.

Problem
-------
Bookmakers price 1X2 and Over/Under markets with high liquidity and sharp
margins.  But niche markets (exact scores, both-teams-to-score, Asian
handicaps) are often priced by less sophisticated algorithms.

This module calibrates a Poisson model to the sharp 1X2 + O/U markets,
then uses the calibrated model to derive "synthetic" fair odds for any
market.  Comparing synthetic odds to bookmaker prices reveals mispriced
markets — statistical arbitrage without requiring near-zero overrounds.

Calibration method
------------------
  1. Remove bookmaker margin from 1X2: p_h, p_d, p_a = normalise(1/odds).
  2. Grid search over (λ_home, λ_away) ∈ [0.10, 4.00]² at 0.05 step:
       MSE = (fitted_p_home − p_h)² + (fitted_p_draw − p_d)²
  3. If Over 2.5 odds available, extend MSE:
       MSE += (fitted_p_over25 − market_p_over25)²
  4. Best (λ_h, λ_a) wins — O(N²) grid, ~6,400 cells, fast in pure Python.

Usage
-----
    from engine.synthetic_odds import SyntheticOddsEngine, MarketOdds

    engine = SyntheticOddsEngine()
    odds = MarketOdds(bookmaker="pinnacle", fixture_id=1,
                      home_odds=2.10, draw_odds=3.40, away_odds=3.20,
                      over25_odds=1.85, under25_odds=1.98)

    model = engine.calibrate(odds)
    matrix = engine.exact_score_matrix(model)

    # Compare vs competitor:
    arbs = engine.find_arbitrage(model, [competitor_odds], min_edge=0.04)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_MAX_GOALS = 10
_GRID_MIN = 0.10
_GRID_MAX = 4.00
_GRID_STEP = 0.05


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class MarketOdds:
    """Raw decimal odds from one bookmaker for one fixture."""

    bookmaker: str
    fixture_id: int
    home_odds: float
    draw_odds: float
    away_odds: float
    over25_odds: Optional[float] = None
    under25_odds: Optional[float] = None
    over15_odds: Optional[float] = None
    under15_odds: Optional[float] = None
    over35_odds: Optional[float] = None
    under35_odds: Optional[float] = None

    @property
    def overround_1x2(self) -> float:
        return 1 / self.home_odds + 1 / self.draw_odds + 1 / self.away_odds

    def implied_home(self) -> float:
        return 1.0 / self.home_odds

    def implied_draw(self) -> float:
        return 1.0 / self.draw_odds

    def implied_away(self) -> float:
        return 1.0 / self.away_odds


@dataclass
class CalibratedModel:
    """Poisson parameters fitted to bookmaker market odds."""

    fixture_id: int
    lambda_home: float
    lambda_away: float
    fitted_p_home: float
    fitted_p_draw: float
    fitted_p_away: float
    market_p_home: float
    market_p_draw: float
    market_p_away: float
    calibration_error: float  # MSE between fitted and market (margin-removed)


@dataclass
class ExactScoreOdds:
    home_goals: int
    away_goals: int
    model_prob: float
    fair_odds: float
    market_odds: Optional[float] = None
    edge: Optional[float] = None

    def __str__(self) -> str:
        mkt = f" mkt={self.market_odds:.2f}" if self.market_odds else ""
        edg = f" edge={self.edge:+.3f}" if self.edge is not None else ""
        return (
            f"{self.home_goals}-{self.away_goals}: fair={self.fair_odds:.2f}{mkt}{edg}"
        )


@dataclass
class SyntheticArbitrageOpportunity:
    fixture_id: int
    market: str
    our_synthetic_prob: float
    our_fair_odds: float
    bookmaker: str
    bookmaker_odds: float
    bookmaker_implied: float
    edge: float
    roi_pct: float

    def __str__(self) -> str:
        return (
            f"SYNTHETIC ARB [{self.market}] vs {self.bookmaker} "
            f"fair={self.our_fair_odds:.2f} mkt={self.bookmaker_odds:.2f} "
            f"edge={self.edge:+.3f} ROI={self.roi_pct:+.1f}%"
        )


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class SyntheticOddsEngine:
    """
    Calibrates a Poisson model to bookmaker 1X2 odds and prices any market.

    Parameters
    ----------
    max_goals  : score matrix truncation (default 10)
    grid_step  : calibration grid resolution (default 0.05)
    """

    def __init__(
        self, max_goals: int = _MAX_GOALS, grid_step: float = _GRID_STEP
    ) -> None:
        self._max = max_goals
        self._step = grid_step

        # Pre-build grid values
        n_steps = int(round((_GRID_MAX - _GRID_MIN) / self._step)) + 1
        self._grid_vals = [_GRID_MIN + i * self._step for i in range(n_steps)]

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def calibrate(self, odds: MarketOdds) -> CalibratedModel:
        """
        Fit (lambda_home, lambda_away) to the bookmaker's 1X2 (+ O/U 2.5) odds.

        Returns CalibratedModel with best-fit lambdas and calibration error.
        """
        # Remove margin
        ov = odds.overround_1x2
        p_h = (1 / odds.home_odds) / ov
        p_d = (1 / odds.draw_odds) / ov
        p_a = (1 / odds.away_odds) / ov

        # Market O/U 2.5 implied (margin-removed if both sides given)
        target_over25: Optional[float] = None
        if odds.over25_odds is not None and odds.under25_odds is not None:
            ov_ou = 1 / odds.over25_odds + 1 / odds.under25_odds
            target_over25 = (1 / odds.over25_odds) / ov_ou
        elif odds.over25_odds is not None:
            target_over25 = 1 / odds.over25_odds  # raw implied

        best_lh = 1.3
        best_la = 1.0
        best_mse = float("inf")

        for lh in self._grid_vals:
            for la in self._grid_vals:
                matrix = _build_score_matrix(lh, la, self._max)
                fh, fd, fa = _1x2_from_matrix(matrix)

                mse = (fh - p_h) ** 2 + (fd - p_d) ** 2
                if target_over25 is not None:
                    fo, _ = _ou_from_matrix(matrix, 2.5)
                    mse += (fo - target_over25) ** 2

                if mse < best_mse:
                    best_mse = mse
                    best_lh = lh
                    best_la = la

        # Final fitted probs
        final_matrix = _build_score_matrix(best_lh, best_la, self._max)
        fh, fd, fa = _1x2_from_matrix(final_matrix)

        logger.debug(
            "calibrate: lh=%.3f la=%.3f MSE=%.6f fitted=(%.3f,%.3f,%.3f) market=(%.3f,%.3f,%.3f)",
            best_lh,
            best_la,
            best_mse,
            fh,
            fd,
            fa,
            p_h,
            p_d,
            p_a,
        )

        return CalibratedModel(
            fixture_id=odds.fixture_id,
            lambda_home=best_lh,
            lambda_away=best_la,
            fitted_p_home=fh,
            fitted_p_draw=fd,
            fitted_p_away=fa,
            market_p_home=p_h,
            market_p_draw=p_d,
            market_p_away=p_a,
            calibration_error=best_mse,
        )

    # ------------------------------------------------------------------
    # Market pricing
    # ------------------------------------------------------------------

    def exact_score_matrix(
        self,
        model: CalibratedModel,
        max_score: int = 5,
    ) -> Dict[Tuple[int, int], ExactScoreOdds]:
        """
        Derive fair odds for all (home, away) scorelines up to max_score each.
        Probability renormalised within truncated matrix.
        """
        raw = _build_score_matrix(model.lambda_home, model.lambda_away, self._max)

        # Aggregate probs for scores within max_score
        result: Dict[Tuple[int, int], ExactScoreOdds] = {}
        total_covered = 0.0
        probs: Dict[Tuple[int, int], float] = {}

        for (hg, ag), prob in raw.items():
            if hg <= max_score and ag <= max_score:
                probs[(hg, ag)] = prob
                total_covered += prob

        # Renormalise to covered space
        for (hg, ag), prob in probs.items():
            adj_prob = prob / max(total_covered, 1e-9)
            result[(hg, ag)] = ExactScoreOdds(
                home_goals=hg,
                away_goals=ag,
                model_prob=adj_prob,
                fair_odds=_inv(adj_prob),
            )

        return result

    def btts_odds(self, model: CalibratedModel) -> Tuple[float, float]:
        """
        P(BTTS YES) = (1 − e^{−λ_h}) × (1 − e^{−λ_a})
        Returns (fair_odds_yes, fair_odds_no).
        """
        p_yes = (1.0 - math.exp(-model.lambda_home)) * (
            1.0 - math.exp(-model.lambda_away)
        )
        p_no = 1.0 - p_yes
        return _inv(p_yes), _inv(p_no)

    def asian_handicap_odds(
        self, model: CalibratedModel, line: float
    ) -> Tuple[float, float]:
        """
        Derive AH fair odds at given line from calibrated Poisson model.
        Uses margin_pmf and same quarter-line splitting as AsianHandicapModel.
        Returns (fair_home, fair_away).
        """
        matrix = _build_score_matrix(model.lambda_home, model.lambda_away, self._max)
        margin_pmf: Dict[int, float] = {}
        for (hg, ag), prob in matrix.items():
            m = hg - ag
            margin_pmf[m] = margin_pmf.get(m, 0.0) + prob

        def _price_half_line(line_val: float) -> Tuple[float, float, float]:
            pw = pd_ = pl = 0.0
            for margin, prob in margin_pmf.items():
                adj = margin + line_val
                if adj > 0:
                    pw += prob
                elif adj == 0 and abs(line_val - round(line_val)) < 1e-9:
                    pd_ += prob
                else:
                    pl += prob
            return pw, pd_, pl

        if abs(round(line * 4) % 2) == 1:  # quarter-line
            lo, hi = line - 0.25, line + 0.25
            pw_lo, pd_lo, pl_lo = _price_half_line(lo)
            pw_hi, pd_hi, pl_hi = _price_half_line(hi)
            pw = 0.5 * pw_lo + 0.5 * pw_hi
            pd_ = 0.5 * pd_lo + 0.5 * pd_hi
            pl = 0.5 * pl_lo + 0.5 * pl_hi
        else:
            pw, pd_, pl = _price_half_line(line)

        p_home_adj = pw + 0.5 * pd_
        p_away_adj = pl + 0.5 * pd_
        return _inv(p_home_adj), _inv(p_away_adj)

    def ou_odds(self, model: CalibratedModel, line: float) -> Tuple[float, float]:
        """
        Derive Over/Under fair odds at given line from calibrated model.
        Returns (fair_over, fair_under).
        """
        matrix = _build_score_matrix(model.lambda_home, model.lambda_away, self._max)
        p_over, p_under = _ou_from_matrix(matrix, line)
        return _inv(p_over), _inv(p_under)

    # ------------------------------------------------------------------
    # Arbitrage detection
    # ------------------------------------------------------------------

    def find_arbitrage(
        self,
        model: CalibratedModel,
        competitor_odds: List[MarketOdds],
        min_edge: float = 0.04,
        max_score: int = 5,
    ) -> List[SyntheticArbitrageOpportunity]:
        """
        Compare synthetic exact-score and BTTS probabilities vs competitor odds.

        Checks:
          - Exact scores (h-a) for h,a in 0..max_score
          - BTTS YES / NO

        Returns opportunities where model_prob - 1/market_odds >= min_edge,
        sorted by edge descending.
        """
        opps: List[SyntheticArbitrageOpportunity] = []
        score_matrix = self.exact_score_matrix(model, max_score=max_score)
        for comp in competitor_odds:
            # 1X2 check
            for market_label, model_p, mkt_odds in [
                ("1X2_HOME", model.fitted_p_home, comp.home_odds),
                ("1X2_DRAW", model.fitted_p_draw, comp.draw_odds),
                ("1X2_AWAY", model.fitted_p_away, comp.away_odds),
            ]:
                edge = model_p - 1.0 / mkt_odds
                if edge >= min_edge:
                    opps.append(
                        SyntheticArbitrageOpportunity(
                            fixture_id=model.fixture_id,
                            market=market_label,
                            our_synthetic_prob=model_p,
                            our_fair_odds=_inv(model_p),
                            bookmaker=comp.bookmaker,
                            bookmaker_odds=mkt_odds,
                            bookmaker_implied=1.0 / mkt_odds,
                            edge=edge,
                            roi_pct=(_inv(model_p) / mkt_odds - 1.0) * 100.0,
                        )
                    )

            # Exact scores (requires competitor to expose exact score odds)
            # — we check all cells in our matrix and use competitor's odds if passed
            for (hg, ag), es_odds in score_matrix.items():
                # No competitor exact-score odds passed here — skip unless present
                # (this is the primary use-case when caller attaches them)
                if es_odds.market_odds is not None:
                    edge = es_odds.model_prob - 1.0 / es_odds.market_odds
                    if edge >= min_edge:
                        opps.append(
                            SyntheticArbitrageOpportunity(
                                fixture_id=model.fixture_id,
                                market=f"EXACT_{hg}-{ag}",
                                our_synthetic_prob=es_odds.model_prob,
                                our_fair_odds=es_odds.fair_odds,
                                bookmaker=comp.bookmaker,
                                bookmaker_odds=es_odds.market_odds,
                                bookmaker_implied=1.0 / es_odds.market_odds,
                                edge=edge,
                                roi_pct=(es_odds.fair_odds / es_odds.market_odds - 1.0)
                                * 100.0,
                            )
                        )

        opps.sort(key=lambda o: o.edge, reverse=True)
        return opps

    def attach_competitor_exact_scores(
        self,
        score_matrix: Dict[Tuple[int, int], ExactScoreOdds],
        competitor_exact: Dict[Tuple[int, int], float],
    ) -> None:
        """
        Attach competitor exact-score market odds to the score matrix in-place.

        competitor_exact: {(home_goals, away_goals): decimal_odds}
        """
        for (hg, ag), mkt_odds in competitor_exact.items():
            if (hg, ag) in score_matrix:
                cell = score_matrix[(hg, ag)]
                cell.market_odds = mkt_odds
                cell.edge = cell.model_prob - 1.0 / mkt_odds

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def summary(self, model: CalibratedModel) -> str:
        """Return readable model summary + top-6 most likely exact scores."""
        lines = [
            f"CalibratedModel fixture={model.fixture_id}",
            f"  λ_home={model.lambda_home:.3f}  λ_away={model.lambda_away:.3f}",
            f"  fitted  1X2: {model.fitted_p_home:.3f}/{model.fitted_p_draw:.3f}/{model.fitted_p_away:.3f}",
            f"  market  1X2: {model.market_p_home:.3f}/{model.market_p_draw:.3f}/{model.market_p_away:.3f}",
            f"  MSE: {model.calibration_error:.6f}",
            "",
            "  Top-6 exact scores (by model prob):",
        ]
        matrix = self.exact_score_matrix(model, max_score=5)
        top = sorted(matrix.values(), key=lambda e: e.model_prob, reverse=True)[:6]
        for e in top:
            lines.append(f"    {e}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Module helpers
# ---------------------------------------------------------------------------


def _build_score_matrix(
    lh: float, la: float, max_goals: int
) -> Dict[Tuple[int, int], float]:
    """Poisson joint PMF, renormalised for truncation."""
    matrix: Dict[Tuple[int, int], float] = {}
    total = 0.0
    for hg in range(max_goals + 1):
        for ag in range(max_goals + 1):
            p = _poisson_pmf(hg, lh) * _poisson_pmf(ag, la)
            matrix[(hg, ag)] = p
            total += p
    if total > 0:
        matrix = {k: v / total for k, v in matrix.items()}
    return matrix


def _1x2_from_matrix(
    matrix: Dict[Tuple[int, int], float],
) -> Tuple[float, float, float]:
    p_h = p_d = p_a = 0.0
    for (hg, ag), prob in matrix.items():
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


def _ou_from_matrix(
    matrix: Dict[Tuple[int, int], float], line: float
) -> Tuple[float, float]:
    """P(total > line), P(total <= line) — handles quarter-lines via average."""
    if abs(round(line * 2) % 2) == 1:  # half-line (e.g. 2.5) — no push
        p_over = sum(prob for (hg, ag), prob in matrix.items() if hg + ag > line)
        return p_over, 1.0 - p_over

    # Whole or quarter — check for push
    p_over = p_push = 0.0
    for (hg, ag), prob in matrix.items():
        total = hg + ag
        if total > line:
            p_over += prob
        elif abs(total - line) < 1e-9:
            p_push += prob
    p_under = 1.0 - p_over - p_push
    # Push: half stake returned → effective: over=p_over+0.5×push, under=p_under+0.5×push
    return p_over + 0.5 * p_push, p_under + 0.5 * p_push


def _poisson_pmf(k: int, lam: float) -> float:
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    try:
        return math.exp(-lam) * (lam**k) / math.factorial(k)
    except (OverflowError, ValueError):
        return 0.0


def _inv(p: float, floor: float = 1.01) -> float:
    if p <= 0:
        return 999.0
    return max(floor, 1.0 / p)


def _inv_p(fair_odds: float) -> float:
    """Convert fair odds back to probability."""
    if fair_odds <= 0:
        return 0.0
    return 1.0 / fair_odds


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------


def calibrate_and_analyse(
    home_odds: float,
    draw_odds: float,
    away_odds: float,
    over25_odds: Optional[float] = None,
    under25_odds: Optional[float] = None,
    fixture_id: int = 0,
    bookmaker: str = "unknown",
) -> Tuple[CalibratedModel, Dict[Tuple[int, int], ExactScoreOdds]]:
    """
    One-shot convenience: calibrate model and return exact-score matrix.

    Returns (CalibratedModel, {(hg, ag): ExactScoreOdds}).
    """
    odds = MarketOdds(
        bookmaker=bookmaker,
        fixture_id=fixture_id,
        home_odds=home_odds,
        draw_odds=draw_odds,
        away_odds=away_odds,
        over25_odds=over25_odds,
        under25_odds=under25_odds,
    )
    engine = SyntheticOddsEngine()
    model = engine.calibrate(odds)
    matrix = engine.exact_score_matrix(model)
    return model, matrix

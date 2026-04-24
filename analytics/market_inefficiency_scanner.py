"""
Market Inefficiency Scanner.

Identifies and classifies systematic pricing errors in bookmaker odds by
comparing model-implied probabilities against sharp-book consensus.

Inefficiency types
------------------
TYPE_A — Favourite-longshot bias:
    Bookmakers systematically overprice longshots (high odds) and
    underprice favourites.  Detected via probability-decile regression
    slope significantly > 1.

TYPE_B — Home-advantage mispricing:
    Bookmaker implied home probability consistently above (or below)
    the model estimate for a given league / venue class.

TYPE_C — Draw-aversion:
    Draws are persistently underpriced in certain league clusters;
    the scanner flags when implied draw probability < model by > threshold.

TYPE_D — Late-movement inefficiency:
    Sharp money moves closing odds away from opening; this module
    quantifies residual mispricing after accounting for steam.

Statistical framework
---------------------
- Z-score for each observation: z = (p_model − p_implied) / σ_pooled
  where σ_pooled = √(p̄(1−p̄) / n_effective)
- Significance threshold: |z| ≥ 1.96  (two-sided 95 %)
- Effect size: Cohen's h = 2 arcsin(√p_model) − 2 arcsin(√p_implied)
- Temporal decay: recent observations weighted by exp(−λ Δt)
  λ = 0.005 per hour (half-life ≈ 6 days)

Usage
-----
    from analytics.market_inefficiency_scanner import MarketInefficiencyScanner

    scanner = MarketInefficiencyScanner()
    results = scanner.scan(predictions)   # list[dict] with model + implied probs
    report  = scanner.report(results)     # summary by type and league
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

_Z_THRESHOLD = 1.96  # 95 % two-sided
_EFFECT_THRESHOLD = 0.05  # Cohen's h minimum for practical significance
_DECAY_LAMBDA = 0.005 / 3600  # per second → half-life ≈ 6 days
_MIN_SAMPLE = 10  # minimum observations for significance test
_DRAW_AVERSION_MIN = 0.05  # minimum implied−model gap for TYPE_C

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class Inefficiency:
    """A single detected market inefficiency observation."""

    fixture_id: str
    league: str
    market: str  # "home" | "draw" | "away"
    inefficiency_type: str  # "TYPE_A" .. "TYPE_D"
    p_model: float  # model probability
    p_implied: float  # bookmaker implied (vig-removed)
    gap: float  # p_model − p_implied (positive = model thinks more likely)
    z_score: float  # standardised effect
    cohen_h: float  # practical effect size
    significant: bool  # |z| ≥ threshold AND |h| ≥ effect threshold
    edge_pct: float  # (p_model / p_implied − 1) × 100  in percent
    timestamp: float  # Unix epoch of observation
    weight: float = 1.0  # temporal decay weight (set after construction)

    def as_dict(self) -> Dict:
        return {
            "fixture_id": self.fixture_id,
            "league": self.league,
            "market": self.market,
            "inefficiency_type": self.inefficiency_type,
            "p_model": round(self.p_model, 6),
            "p_implied": round(self.p_implied, 6),
            "gap": round(self.gap, 6),
            "z_score": round(self.z_score, 4),
            "cohen_h": round(self.cohen_h, 4),
            "significant": self.significant,
            "edge_pct": round(self.edge_pct, 4),
            "weight": round(self.weight, 4),
        }


@dataclass
class LeagueInefficiencyProfile:
    """Aggregated inefficiency statistics for one league."""

    league: str
    n_observations: int
    weighted_n: float
    mean_gap: float  # weighted mean of (p_model − p_implied)
    std_gap: float
    z_score: float  # z-test on the mean gap
    significant: bool
    type_counts: Dict[str, int] = field(default_factory=dict)
    dominant_type: str = ""
    mean_edge_pct: float = 0.0


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------


def _implied_prob(odds: float) -> float:
    """Inverse odds, clamped to valid range."""
    return 1.0 / max(1.01, odds)


def _remove_vig(
    home_odds: float, draw_odds: float, away_odds: float
) -> Tuple[float, float, float]:
    """Shin-method vig removal (approximate: normalise to sum-to-1)."""
    ih = _implied_prob(home_odds)
    id_ = _implied_prob(draw_odds)
    ia = _implied_prob(away_odds)
    total = ih + id_ + ia
    if total <= 0:
        return 1 / 3, 1 / 3, 1 / 3
    return ih / total, id_ / total, ia / total


def _z_score(p_model: float, p_implied: float, n_effective: float) -> float:
    """Z-score for difference between model and implied probability."""
    p_bar = (p_model + p_implied) / 2.0
    var = p_bar * (1.0 - p_bar) / max(n_effective, 1.0)
    se = math.sqrt(var) if var > 0 else 1e-9
    return (p_model - p_implied) / se


def _cohen_h(p1: float, p2: float) -> float:
    """Cohen's h effect size for two proportions."""
    phi1 = 2.0 * math.asin(math.sqrt(max(0.0, min(1.0, p1))))
    phi2 = 2.0 * math.asin(math.sqrt(max(0.0, min(1.0, p2))))
    return phi1 - phi2


def _decay_weight(ts: float, now: float) -> float:
    """Exponential decay weight based on observation age."""
    delta = max(0.0, now - ts)
    return math.exp(-_DECAY_LAMBDA * delta)


def _edge_pct(p_model: float, p_implied: float) -> float:
    """Model edge over implied in percent."""
    if p_implied <= 0:
        return 0.0
    return (p_model / p_implied - 1.0) * 100.0


# ---------------------------------------------------------------------------
# Inefficiency classifiers
# ---------------------------------------------------------------------------


def _classify_type(
    market: str,
    p_model: float,
    p_implied: float,
    gap: float,
) -> str:
    """
    Classify inefficiency type based on market and direction.

    TYPE_A — favourite-longshot: home favourite (high p) underpriced
    TYPE_B — home-advantage: home market gap regardless of direction
    TYPE_C — draw aversion: draw systematically underpriced
    TYPE_D — general residual
    """
    if market == "draw":
        if gap > _DRAW_AVERSION_MIN:
            return "TYPE_C"  # draw underpriced by bookmaker
        return "TYPE_D"

    if market == "home":
        if p_implied >= 0.50 and gap > 0:
            return "TYPE_A"  # home favourite underpriced
        if p_implied < 0.30 and gap < 0:
            return "TYPE_A"  # home longshot overpriced
        return "TYPE_B"

    if market == "away":
        if p_implied >= 0.40 and gap > 0:
            return "TYPE_A"
        if p_implied < 0.20 and gap < 0:
            return "TYPE_A"
        return "TYPE_D"

    return "TYPE_D"


# ---------------------------------------------------------------------------
# Scanner
# ---------------------------------------------------------------------------


class MarketInefficiencyScanner:
    """
    Detects and classifies systematic bookmaker pricing errors.

    Parameters
    ----------
    z_threshold      : z-score cutoff for statistical significance (default 1.96)
    effect_threshold : Cohen's h cutoff for practical significance (default 0.05)
    decay_lambda     : temporal decay rate per second (default 0.005/3600)
    min_sample       : minimum observations per league for significance (default 10)
    """

    def __init__(
        self,
        z_threshold: float = _Z_THRESHOLD,
        effect_threshold: float = _EFFECT_THRESHOLD,
        decay_lambda: float = _DECAY_LAMBDA,
        min_sample: int = _MIN_SAMPLE,
    ) -> None:
        self._z_thresh = z_threshold
        self._h_thresh = effect_threshold
        self._decay = decay_lambda
        self._min_sample = min_sample
        # Running history for league profiling
        self._history: List[Inefficiency] = []

    # ------------------------------------------------------------------
    # Primary scan
    # ------------------------------------------------------------------

    def scan(self, predictions: List[Dict]) -> List[Dict]:
        """
        Scan a list of predictions for market inefficiencies.

        Each prediction must contain:
          - home_win, draw, away_win  : model probabilities
          - home_odds, draw_odds, away_odds  : bookmaker odds
        Optional:
          - fixture_id, league, timestamp

        Returns the input list enriched with inefficiency fields.
        """
        now = time.time()
        results = []

        for pred in predictions:
            enriched = dict(pred)
            try:
                ineffs = self._scan_one(pred, now)
                if ineffs:
                    best = max(ineffs, key=lambda x: abs(x.z_score))
                    best.weight = _decay_weight(best.timestamp, now)
                    self._history.append(best)
                    enriched.update(best.as_dict())
                    enriched["all_inefficiencies"] = [i.as_dict() for i in ineffs]
                else:
                    enriched["inefficiency_type"] = "NONE"
                    enriched["z_score"] = 0.0
                    enriched["significant"] = False
                    enriched["edge_pct"] = 0.0
                    enriched["cohen_h"] = 0.0
            except Exception as exc:
                logger.debug("MarketInefficiencyScanner.scan: error on record: %s", exc)
            results.append(enriched)

        return results

    def _scan_one(self, pred: Dict, now: float) -> List[Inefficiency]:
        """Detect inefficiencies for a single prediction dict."""
        p_h_model = float(pred.get("home_win", pred.get("home", 0.0)))
        p_d_model = float(pred.get("draw", 0.0))
        p_a_model = float(pred.get("away_win", pred.get("away", 0.0)))

        home_odds = float(pred.get("home_odds", 0.0))
        draw_odds = float(pred.get("draw_odds", 0.0))
        away_odds = float(pred.get("away_odds", 0.0))

        if home_odds <= 1.0 or draw_odds <= 1.0 or away_odds <= 1.0:
            return []

        p_h_imp, p_d_imp, p_a_imp = _remove_vig(home_odds, draw_odds, away_odds)

        fixture_id = str(pred.get("fixture_id", pred.get("match_id", "")))
        league = str(pred.get("league", "Unknown"))
        ts = float(pred.get("timestamp", now))
        n_eff = max(1.0, float(pred.get("n_history", 1.0)))

        inefficiencies = []
        for market, p_model, p_implied in [
            ("home", p_h_model, p_h_imp),
            ("draw", p_d_model, p_d_imp),
            ("away", p_a_model, p_a_imp),
        ]:
            if p_model <= 0 or p_implied <= 0:
                continue
            gap = p_model - p_implied
            z = _z_score(p_model, p_implied, n_eff)
            h = _cohen_h(p_model, p_implied)
            sig = abs(z) >= self._z_thresh and abs(h) >= self._h_thresh
            itype = _classify_type(market, p_model, p_implied, gap)
            edge = _edge_pct(p_model, p_implied)

            ineff = Inefficiency(
                fixture_id=fixture_id,
                league=league,
                market=market,
                inefficiency_type=itype,
                p_model=p_model,
                p_implied=p_implied,
                gap=gap,
                z_score=z,
                cohen_h=h,
                significant=sig,
                edge_pct=edge,
                timestamp=ts,
            )
            inefficiencies.append(ineff)

        return [i for i in inefficiencies if abs(i.gap) > 0.001]

    # ------------------------------------------------------------------
    # Aggregate report
    # ------------------------------------------------------------------

    def report(self, results: Optional[List[Dict]] = None) -> Dict:
        """
        Produce a league-level inefficiency report.

        Uses scan results if provided, otherwise uses the accumulated history.
        """
        now = time.time()
        history = self._history if not results else self._extract_history(results, now)

        if not history:
            return {"n_observations": 0, "leagues": {}}

        # Apply temporal weights
        for ineff in history:
            ineff.weight = _decay_weight(ineff.timestamp, now)

        # Group by league
        by_league: Dict[str, List[Inefficiency]] = {}
        for ineff in history:
            by_league.setdefault(ineff.league, []).append(ineff)

        leagues_out = {}
        for league, items in by_league.items():
            profile = self._profile_league(league, items)
            leagues_out[league] = {
                "n": profile.n_observations,
                "weighted_n": round(profile.weighted_n, 2),
                "mean_gap": round(profile.mean_gap, 6),
                "std_gap": round(profile.std_gap, 6),
                "z_score": round(profile.z_score, 4),
                "significant": profile.significant,
                "dominant_type": profile.dominant_type,
                "mean_edge_pct": round(profile.mean_edge_pct, 4),
                "type_counts": profile.type_counts,
            }

        # Global type distribution
        type_totals: Dict[str, int] = {}
        for ineff in history:
            type_totals[ineff.inefficiency_type] = (
                type_totals.get(ineff.inefficiency_type, 0) + 1
            )

        sig_count = sum(1 for i in history if i.significant)
        return {
            "n_observations": len(history),
            "n_significant": sig_count,
            "significance_rate": round(sig_count / len(history), 4) if history else 0.0,
            "type_distribution": type_totals,
            "leagues": leagues_out,
        }

    def _profile_league(
        self, league: str, items: List[Inefficiency]
    ) -> LeagueInefficiencyProfile:
        n = len(items)
        total_w = sum(i.weight for i in items)

        if total_w == 0:
            total_w = 1.0

        w_mean_gap = sum(i.gap * i.weight for i in items) / total_w
        w_mean_edge = sum(i.edge_pct * i.weight for i in items) / total_w
        variance = sum(i.weight * (i.gap - w_mean_gap) ** 2 for i in items) / total_w
        std_gap = math.sqrt(variance) if variance > 0 else 0.0

        # Weighted z-test on mean gap
        se = std_gap / math.sqrt(max(1, n))
        z = w_mean_gap / se if se > 0 else 0.0
        significant = abs(z) >= self._z_thresh and n >= self._min_sample

        type_counts: Dict[str, int] = {}
        for i in items:
            type_counts[i.inefficiency_type] = (
                type_counts.get(i.inefficiency_type, 0) + 1
            )

        dominant = max(type_counts, key=type_counts.get) if type_counts else ""

        return LeagueInefficiencyProfile(
            league=league,
            n_observations=n,
            weighted_n=round(total_w, 2),
            mean_gap=w_mean_gap,
            std_gap=std_gap,
            z_score=z,
            significant=significant,
            type_counts=type_counts,
            dominant_type=dominant,
            mean_edge_pct=w_mean_edge,
        )

    def _extract_history(self, results: List[Dict], now: float) -> List[Inefficiency]:
        """Re-build Inefficiency objects from scan result dicts."""
        out = []
        for r in results:
            if "gap" not in r or "market" not in r:
                continue
            ineff = Inefficiency(
                fixture_id=str(r.get("fixture_id", "")),
                league=str(r.get("league", "Unknown")),
                market=str(r.get("market", "home")),
                inefficiency_type=str(r.get("inefficiency_type", "TYPE_D")),
                p_model=float(r.get("p_model", 0.0)),
                p_implied=float(r.get("p_implied", 0.0)),
                gap=float(r.get("gap", 0.0)),
                z_score=float(r.get("z_score", 0.0)),
                cohen_h=float(r.get("cohen_h", 0.0)),
                significant=bool(r.get("significant", False)),
                edge_pct=float(r.get("edge_pct", 0.0)),
                timestamp=float(r.get("timestamp", now)),
                weight=float(r.get("weight", 1.0)),
            )
            out.append(ineff)
        return out

    # ------------------------------------------------------------------
    # Favourite-longshot bias analysis
    # ------------------------------------------------------------------

    def favourite_longshot_analysis(self, predictions: List[Dict]) -> Dict:
        """
        Regression of actual win rate on implied probability deciles.

        A slope > 1 indicates longshot overpricing (classic FLB).
        Returns regression slope, intercept, and R².
        """
        pairs: List[Tuple[float, float]] = []
        for pred in predictions:
            home_odds = float(pred.get("home_odds", 0.0))
            actual = float(pred.get("actual_home_win", -1))
            if home_odds <= 1.0 or actual < 0:
                continue
            p_imp, _, _ = _remove_vig(
                home_odds,
                float(pred.get("draw_odds", 3.5)),
                float(pred.get("away_odds", 3.5)),
            )
            pairs.append((p_imp, actual))

        if len(pairs) < 10:
            return {"slope": None, "intercept": None, "r2": None, "n": len(pairs)}

        xs = [p for p, _ in pairs]
        ys = [y for _, y in pairs]
        n = len(xs)
        mx = sum(xs) / n
        my = sum(ys) / n
        ssxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        ssxx = sum((x - mx) ** 2 for x in xs)
        slope = ssxy / ssxx if ssxx > 0 else 0.0
        intercept = my - slope * mx
        yhat = [slope * x + intercept for x in xs]
        ss_res = sum((y - yh) ** 2 for y, yh in zip(ys, yhat))
        ss_tot = sum((y - my) ** 2 for y in ys)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        return {
            "slope": round(slope, 4),
            "intercept": round(intercept, 4),
            "r2": round(r2, 4),
            "n": n,
            "flb_detected": slope > 1.1,
        }

    # ------------------------------------------------------------------
    # History management
    # ------------------------------------------------------------------

    def clear_history(self) -> None:
        self._history.clear()

    def history_summary(self) -> Dict:
        return {
            "total": len(self._history),
            "significant": sum(1 for i in self._history if i.significant),
            "leagues": list({i.league for i in self._history}),
        }

"""
League Predictability Analyser.

Quantifies how predictable a football league is using proper scoring rules
and structural balance metrics.

Metrics
-------
1. **Ranked Probability Score (RPS)**
   Proper scoring rule for ordinal outcomes.
   RPS = ½ × Σ_{k=1}^{K-1} (F_k − O_k)²
   where F_k = cumulative predicted prob up to outcome k,
         O_k = cumulative observed (indicator).
   Lower RPS → better calibration.  Reference: Constantinou & Fenton (2012).

2. **Brier Score Decomposition** (Murphy 1973)
   BS = REL − RES + UNC
   - REL (reliability): how much forecast probabilities deviate from
     observed frequencies within probability bins.
   - RES (resolution): how much bin frequencies deviate from climatology.
   - UNC (uncertainty): base-rate uncertainty = p̄(1−p̄).

3. **Competitive Balance Index (CBI)**
   CBI = 1 − Gini(points_distribution)
   Range [0,1]; 1 = perfectly equal, 0 = monopoly.
   Derived from Depken (1999) concentration measure.

4. **Upset Rate**
   Fraction of matches where the implied favourite (home_odds < away_odds)
   actually lost.

5. **Home Advantage Factor (HAF)**
   HAF = home_win_rate / (home_win_rate + away_win_rate)
   Controls for draw frequency.

6. **Predictability Score** (composite)
   P = w1×(1−RPS_norm) + w2×(1−BS_norm) + w3×CBI_norm
   where norms map each metric to [0,1].

Usage
-----
    from analytics.league_predictability import LeaguePredictabilityAnalyser

    analyser = LeaguePredictabilityAnalyser()
    report = analyser.analyse(league_records)  # list[dict]
    summary = analyser.summary_table(all_leagues_records)
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_RPS_OUTCOMES = ["home_win", "draw", "away_win"]  # ordered for RPS
_BRIER_BINS = 10
_W_RPS = 0.40
_W_BRIER = 0.35
_W_CBI = 0.25


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class BrierDecomposition:
    """Murphy (1973) decomposition of the Brier score."""

    brier_score: float
    reliability: float  # REL — lower is better (calibration error)
    resolution: float  # RES — higher is better (sharpness)
    uncertainty: float  # UNC — inherent difficulty; independent of forecasts
    skill_score: float  # BSS = 1 − BS/BS_clim = RES/UNC − REL/UNC

    def as_dict(self) -> Dict:
        return {
            "brier_score": round(self.brier_score, 6),
            "reliability": round(self.reliability, 6),
            "resolution": round(self.resolution, 6),
            "uncertainty": round(self.uncertainty, 6),
            "skill_score": round(self.skill_score, 4),
        }


@dataclass
class LeagueReport:
    """Full predictability report for one league."""

    league: str
    n_matches: int
    rps: float  # mean Ranked Probability Score
    rps_skill: float  # RPS relative to climatology
    brier: BrierDecomposition
    home_win_rate: float
    draw_rate: float
    away_win_rate: float
    home_advantage_factor: float  # HAF
    upset_rate: float
    competitive_balance_index: float  # CBI
    predictability_score: float  # composite [0,1], higher = more predictable
    grade: str  # A/B/C/D

    def as_dict(self) -> Dict:
        return {
            "league": self.league,
            "n_matches": self.n_matches,
            "rps": round(self.rps, 6),
            "rps_skill": round(self.rps_skill, 4),
            **{f"brier_{k}": v for k, v in self.brier.as_dict().items()},
            "home_win_rate": round(self.home_win_rate, 4),
            "draw_rate": round(self.draw_rate, 4),
            "away_win_rate": round(self.away_win_rate, 4),
            "home_advantage_factor": round(self.home_advantage_factor, 4),
            "upset_rate": round(self.upset_rate, 4),
            "competitive_balance_index": round(self.competitive_balance_index, 4),
            "predictability_score": round(self.predictability_score, 4),
            "grade": self.grade,
        }


# ---------------------------------------------------------------------------
# RPS helpers
# ---------------------------------------------------------------------------


def _ranked_prob_score(
    p_home: float,
    p_draw: float,
    p_away: float,
    outcome: str,  # "home_win" | "draw" | "away_win"
) -> float:
    """
    Ranked Probability Score for a single 3-outcome prediction.
    RPS ∈ [0, 1]; lower is better.
    """
    outcome_idx = {"home_win": 0, "draw": 1, "away_win": 2}.get(outcome, -1)
    if outcome_idx < 0:
        return 0.5  # unknown outcome → maximum entropy

    probs = [p_home, p_draw, p_away]
    # Cumulative predicted probabilities
    cum_f = [sum(probs[: k + 1]) for k in range(len(probs))]
    # Cumulative observed
    cum_o = [1.0 if k >= outcome_idx else 0.0 for k in range(len(probs))]
    # RPS = ½ Σ_{k=1}^{K-1} (F_k − O_k)²
    rps = 0.5 * sum((cum_f[k] - cum_o[k]) ** 2 for k in range(len(probs) - 1))
    return rps


def _climatological_rps(home_rate: float, draw_rate: float, away_rate: float) -> float:
    """RPS when always predicting the climatological frequencies."""
    total_rps = 0.0
    weights = [home_rate, draw_rate, away_rate]
    for outcome_idx in range(3):
        cum_o = [1.0 if k >= outcome_idx else 0.0 for k in range(3)]
        cum_f = [sum(weights[: k + 1]) for k in range(3)]
        rps_i = 0.5 * sum((cum_f[k] - cum_o[k]) ** 2 for k in range(2))
        total_rps += weights[outcome_idx] * rps_i
    return total_rps


# ---------------------------------------------------------------------------
# Brier decomposition helpers
# ---------------------------------------------------------------------------


def _brier_decompose(
    pred_probs: List[float],
    actuals: List[int],  # 0 or 1
    n_bins: int = _BRIER_BINS,
) -> BrierDecomposition:
    """
    Murphy (1973) Brier score decomposition for a binary outcome.

    pred_probs : list of forecast probabilities [0, 1]
    actuals    : list of observed outcomes (0 or 1)
    """
    n = len(pred_probs)
    if n == 0:
        return BrierDecomposition(0.0, 0.0, 0.0, 0.0, 0.0)

    bs = sum((p - o) ** 2 for p, o in zip(pred_probs, actuals)) / n
    clim = sum(actuals) / n  # climatological frequency
    unc = clim * (1.0 - clim)

    # Bin predictions
    bin_width = 1.0 / n_bins
    bins: Dict[int, Tuple[float, float, int]] = {}  # bin → (sum_p, sum_o, count)
    for p, o in zip(pred_probs, actuals):
        b = min(int(p / bin_width), n_bins - 1)
        s_p, s_o, cnt = bins.get(b, (0.0, 0.0, 0))
        bins[b] = (s_p + p, s_o + o, cnt + 1)

    rel = 0.0
    res = 0.0
    for s_p, s_o, cnt in bins.values():
        if cnt == 0:
            continue
        bar_p = s_p / cnt
        bar_o = s_o / cnt
        rel += cnt * (bar_p - bar_o) ** 2
        res += cnt * (bar_o - clim) ** 2

    rel /= n
    res /= n

    skill = (res - rel) / unc if unc > 0 else 0.0

    return BrierDecomposition(
        brier_score=round(bs, 6),
        reliability=round(rel, 6),
        resolution=round(res, 6),
        uncertainty=round(unc, 6),
        skill_score=round(skill, 4),
    )


# ---------------------------------------------------------------------------
# Gini / CBI helper
# ---------------------------------------------------------------------------


def _gini(values: List[float]) -> float:
    """Gini coefficient of a list of non-negative values."""
    if not values:
        return 0.0
    values = sorted(max(0.0, v) for v in values)
    n = len(values)
    total = sum(values) or 1.0
    cumsum = 0.0
    g = 0.0
    for i, v in enumerate(values, start=1):
        cumsum += v
        g += (2 * i - n - 1) * v
    return g / (n * total)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class LeaguePredictabilityAnalyser:
    """
    Computes structural and probabilistic predictability metrics per league.

    Parameters
    ----------
    brier_bins      : number of probability bins for Brier decomposition (default 10)
    w_rps / w_brier / w_cbi : composite score weights (must sum to 1)
    """

    def __init__(
        self,
        brier_bins: int = _BRIER_BINS,
        w_rps: float = _W_RPS,
        w_brier: float = _W_BRIER,
        w_cbi: float = _W_CBI,
    ) -> None:
        self._bins = brier_bins
        self._w = {"rps": w_rps, "brier": w_brier, "cbi": w_cbi}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyse(
        self,
        records: List[Dict],
        league: Optional[str] = None,
    ) -> LeagueReport:
        """
        Produce a full predictability report for a set of match records.

        Each record must contain:
          - actual_outcome : "home_win" | "draw" | "away_win"
          - p_home, p_draw, p_away : model probabilities
        Optional:
          - home_odds, away_odds (for upset rate and HAF)
          - points (team standing points for CBI)
          - league
        """
        if not records:
            return self._empty_report(league or "Unknown")

        league_name = league or str(records[0].get("league", "Unknown"))

        # Collect per-match data
        rps_vals: List[float] = []
        home_probs: List[float] = []
        home_actuals: List[int] = []
        outcomes: List[str] = []
        upsets = 0
        n_decidable = 0
        points_dist: List[float] = []

        for r in records:
            outcome = str(r.get("actual_outcome", ""))
            if outcome not in _RPS_OUTCOMES:
                continue

            ph = float(r.get("p_home", r.get("home_win", 1 / 3)))
            pd_ = float(r.get("p_draw", r.get("draw", 1 / 3)))
            pa = float(r.get("p_away", r.get("away_win", 1 / 3)))

            # Normalise
            total = ph + pd_ + pa
            if total > 0:
                ph, pd_, pa = ph / total, pd_ / total, pa / total

            rps_vals.append(_ranked_prob_score(ph, pd_, pa, outcome))
            home_probs.append(ph)
            home_actuals.append(1 if outcome == "home_win" else 0)
            outcomes.append(outcome)

            # Upset detection
            home_odds = float(r.get("home_odds", 2.0))
            away_odds = float(r.get("away_odds", 2.0))
            if home_odds > 1.0 and away_odds > 1.0:
                n_decidable += 1
                implied_fav = "home" if home_odds < away_odds else "away"
                actual_winner = (
                    "home"
                    if outcome == "home_win"
                    else "away"
                    if outcome == "away_win"
                    else "draw"
                )
                if actual_winner != "draw" and actual_winner != implied_fav:
                    upsets += 1

            # Points for CBI
            pts = r.get("home_points")
            if pts is not None:
                points_dist.append(float(pts))

        n = len(rps_vals)
        if n == 0:
            return self._empty_report(league_name)

        home_rate = outcomes.count("home_win") / n
        draw_rate = outcomes.count("draw") / n
        away_rate = outcomes.count("away_win") / n

        mean_rps = sum(rps_vals) / n
        clim_rps = _climatological_rps(home_rate, draw_rate, away_rate)
        rps_skill = 1.0 - (mean_rps / clim_rps) if clim_rps > 0 else 0.0

        brier = _brier_decompose(home_probs, home_actuals, self._bins)

        haf = (
            home_rate / (home_rate + away_rate) if (home_rate + away_rate) > 0 else 0.5
        )
        upset_rate = upsets / n_decidable if n_decidable > 0 else 0.0

        gini_val = _gini(points_dist) if points_dist else 0.5
        cbi = 1.0 - gini_val

        p_score = self._composite_score(mean_rps, brier.brier_score, cbi)
        grade = self._grade(p_score)

        return LeagueReport(
            league=league_name,
            n_matches=n,
            rps=mean_rps,
            rps_skill=rps_skill,
            brier=brier,
            home_win_rate=home_rate,
            draw_rate=draw_rate,
            away_win_rate=away_rate,
            home_advantage_factor=haf,
            upset_rate=upset_rate,
            competitive_balance_index=cbi,
            predictability_score=p_score,
            grade=grade,
        )

    def summary_table(self, records: List[Dict]) -> List[Dict]:
        """
        Analyse records grouped by league and return a sorted summary table.

        Expects each record to have a 'league' key.
        """
        by_league: Dict[str, List[Dict]] = defaultdict(list)
        for r in records:
            lg = str(r.get("league", "Unknown"))
            by_league[lg].append(r)

        reports = []
        for lg, recs in by_league.items():
            report = self.analyse(recs, league=lg)
            reports.append(report.as_dict())

        return sorted(reports, key=lambda x: x["predictability_score"], reverse=True)

    # ------------------------------------------------------------------
    # Calibration analysis
    # ------------------------------------------------------------------

    def calibration_curve(
        self,
        records: List[Dict],
        market: str = "home_win",
        n_bins: int = 10,
    ) -> List[Dict]:
        """
        Build reliability diagram data: predicted probability vs. observed rate.

        Returns a list of bin dicts: {p_mid, p_model_mean, p_actual, n, gap}.
        """
        key_map = {
            "home_win": ("p_home", "home_win"),
            "draw": ("p_draw", "draw"),
            "away_win": ("p_away", "away_win"),
        }
        prob_key, outcome_val = key_map.get(market, ("p_home", "home_win"))

        width = 1.0 / n_bins
        bins: Dict[int, List[Tuple[float, int]]] = {}

        for r in records:
            outcome = str(r.get("actual_outcome", ""))
            p = float(r.get(prob_key, 0.0))
            actual = 1 if outcome == outcome_val else 0
            b = min(int(p / width), n_bins - 1)
            bins.setdefault(b, []).append((p, actual))

        result = []
        for b in range(n_bins):
            pts = bins.get(b, [])
            p_mid = (b + 0.5) * width
            if not pts:
                result.append(
                    {
                        "p_mid": round(p_mid, 3),
                        "p_model_mean": p_mid,
                        "p_actual": None,
                        "n": 0,
                        "gap": None,
                    }
                )
                continue
            n = len(pts)
            p_mean = sum(p for p, _ in pts) / n
            p_act = sum(a for _, a in pts) / n
            result.append(
                {
                    "p_mid": round(p_mid, 3),
                    "p_model_mean": round(p_mean, 4),
                    "p_actual": round(p_act, 4),
                    "n": n,
                    "gap": round(p_mean - p_act, 4),
                }
            )

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _composite_score(self, rps: float, bs: float, cbi: float) -> float:
        """
        Composite predictability score ∈ [0, 1].

        RPS ∈ [0, 0.5] → normalise to [0, 1] inverted
        BS  ∈ [0, 0.5] → normalise to [0, 1] inverted
        CBI ∈ [0, 1]   → direct (higher = more balanced = harder to predict)
        Note: CBI contribution is negative — balanced leagues are less predictable.
        """
        rps_norm = 1.0 - min(1.0, rps / 0.25)  # 0.25 = expected RPS for random
        bs_norm = 1.0 - min(1.0, bs / 0.25)
        cbi_penalty = 1.0 - cbi  # more balanced → less predictable

        score = (
            self._w["rps"] * rps_norm
            + self._w["brier"] * bs_norm
            + self._w["cbi"] * cbi_penalty
        )
        return round(max(0.0, min(1.0, score)), 4)

    @staticmethod
    def _grade(score: float) -> str:
        if score >= 0.72:
            return "A"
        if score >= 0.58:
            return "B"
        if score >= 0.44:
            return "C"
        return "D"

    @staticmethod
    def _empty_report(league: str) -> LeagueReport:
        empty_brier = BrierDecomposition(0.0, 0.0, 0.0, 0.0, 0.0)
        return LeagueReport(
            league=league,
            n_matches=0,
            rps=0.0,
            rps_skill=0.0,
            brier=empty_brier,
            home_win_rate=0.0,
            draw_rate=0.0,
            away_win_rate=0.0,
            home_advantage_factor=0.5,
            upset_rate=0.0,
            competitive_balance_index=0.5,
            predictability_score=0.0,
            grade="D",
        )

"""
Referee–VAR Pairing Matrix
===========================

Analysing a single referee is amateur-level analysis.  The real edge lies in
the *relationship* between the on-field referee and the VAR official in the
monitor room.  A junior VAR will not contradict a senior referee even when the
on-field decision is clearly wrong — suppressing OFRs and penalty calls.

Authority gap model
-------------------
    authority_score = years_experience × 0.4 + international_matches × 0.01
                      + top_flight_matches × 0.005

    OFR multiplier = 1 / (1 + max(gap, 0) × 0.08)
    (gap > 0 means referee is more senior → fewer OFRs)

Usage
-----
    from engine.var_pairing import RefVARAnalyzer

    analyzer = RefVARAnalyzer()
    analysis = analyzer.analyse_pairing("M. Hartmann", "J. Kovacic",
                                        home_aggression=1.20, away_aggression=0.90)
    print(analysis)
    edges = analyzer.find_edges(analysis, {"penalty_yes": 3.20, "over_45y": 1.90})
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class RefereeRecord:
    """Historical statistics for one referee."""

    name: str
    years_experience: int
    international_matches: int
    top_flight_matches: int
    avg_yellows_per_match: float
    avg_reds_per_match: float
    avg_penalties_awarded: float
    avg_fouls_called: float
    ofr_rate: float  # fraction of matches with at least one OFR (as ref)

    @property
    def authority_score(self) -> float:
        return (
            self.years_experience * 0.4
            + self.international_matches * 0.01
            + self.top_flight_matches * 0.005
        )


@dataclass
class VARRecord:
    """Statistics for one VAR official."""

    name: str
    years_experience: int
    international_matches: int
    top_flight_matches: int
    var_interventions_per_match: float
    ofr_initiated_per_match: float
    penalty_overturned_rate: float

    @property
    def authority_score(self) -> float:
        return (
            self.years_experience * 0.4
            + self.international_matches * 0.01
            + self.top_flight_matches * 0.005
        )


@dataclass
class RefVARPairing:
    referee: RefereeRecord
    var_official: VARRecord
    fixture_id: Optional[int] = None
    home_team: str = ""
    away_team: str = ""
    home_aggression: float = 1.0
    away_aggression: float = 1.0


@dataclass
class PairingAnalysis:
    pairing: RefVARPairing
    authority_gap: float
    ofr_multiplier: float
    expected_penalties: float
    expected_yellows: float
    expected_reds: float
    p_penalty_in_match: float
    p_over_45_yellows: float
    p_over_35_yellows: float
    p_over_05_reds: float
    recommended_markets: List[str]
    notes: List[str]

    def __str__(self) -> str:
        lines = [
            "=" * 56,
            f"REF–VAR PAIRING: {self.pairing.referee.name} / {self.pairing.var_official.name}",
            f"Authority gap: {self.authority_gap:+.2f} (positive = ref is senior)",
            f"OFR multiplier: ×{self.ofr_multiplier:.3f}",
            "-" * 56,
            f"Expected penalties : {self.expected_penalties:.3f}",
            f"Expected yellows   : {self.expected_yellows:.2f}",
            f"Expected reds      : {self.expected_reds:.3f}",
            "-" * 56,
            f"P(penalty)         : {self.p_penalty_in_match:.3f}",
            f"P(over 4.5 yellows): {self.p_over_45_yellows:.3f}",
            f"P(over 3.5 yellows): {self.p_over_35_yellows:.3f}",
            f"P(red card yes)    : {self.p_over_05_reds:.3f}",
            "-" * 56,
            "Recommended markets:",
        ]
        for mkt in self.recommended_markets:
            lines.append(f"  → {mkt}")
        if self.notes:
            lines.append("-" * 56)
            for note in self.notes:
                lines.append(f"  • {note}")
        lines.append("=" * 56)
        return "\n".join(lines)


@dataclass
class PairingEdge:
    market: str
    model_prob: float
    book_odds: float
    book_implied: float
    edge_pct: float
    is_value: bool


# ---------------------------------------------------------------------------
# Built-in database
# ---------------------------------------------------------------------------

_REFEREE_DB: List[RefereeRecord] = [
    RefereeRecord("M. Hartmann", 22, 85, 520, 4.20, 0.12, 0.32, 24.5, 0.28),
    RefereeRecord("J. Kovacic", 18, 62, 380, 3.80, 0.08, 0.28, 22.8, 0.22),
    RefereeRecord("P. Rossi", 28, 140, 720, 4.80, 0.18, 0.45, 26.2, 0.35),
    RefereeRecord("A. Webb", 15, 45, 280, 3.60, 0.07, 0.25, 21.0, 0.20),
    RefereeRecord("C. Taylor", 20, 78, 450, 4.10, 0.10, 0.30, 23.5, 0.25),
    RefereeRecord("K. Friend", 24, 92, 560, 4.50, 0.14, 0.38, 25.0, 0.30),
    RefereeRecord("S. Marciniak", 26, 130, 680, 4.60, 0.15, 0.40, 25.5, 0.33),
    RefereeRecord("D. Makkelie", 19, 70, 420, 3.90, 0.09, 0.27, 22.2, 0.21),
    RefereeRecord("F. Zwayer", 21, 80, 480, 4.30, 0.11, 0.35, 24.0, 0.27),
    RefereeRecord("B. Kuipers", 25, 110, 620, 4.70, 0.16, 0.42, 25.8, 0.32),
    RefereeRecord("C. Cakir", 23, 95, 540, 4.40, 0.13, 0.36, 24.8, 0.29),
    RefereeRecord("R. Brych", 20, 75, 440, 4.00, 0.10, 0.29, 22.8, 0.23),
    RefereeRecord("A. Oliver", 17, 55, 320, 3.70, 0.08, 0.26, 21.5, 0.20),
    RefereeRecord("P. Gago", 14, 40, 260, 3.50, 0.07, 0.23, 20.5, 0.18),
    RefereeRecord("T. Skomina", 24, 100, 580, 4.55, 0.14, 0.39, 25.2, 0.31),
]

_VAR_DB: List[VARRecord] = [
    VARRecord("L. Banti", 20, 70, 450, 2.10, 0.45, 0.18),
    VARRecord("M. Fritz", 16, 50, 320, 1.80, 0.35, 0.14),
    VARRecord("P. Tierney", 12, 30, 220, 1.50, 0.28, 0.10),
    VARRecord("R. East", 8, 15, 140, 1.20, 0.20, 0.08),
    VARRecord("C. Soares", 18, 65, 400, 2.00, 0.42, 0.16),
    VARRecord("A. Marriner", 22, 85, 520, 2.30, 0.50, 0.20),
    VARRecord("G. Scott", 10, 20, 180, 1.35, 0.25, 0.09),
    VARRecord("N. Dingert", 14, 40, 280, 1.65, 0.32, 0.12),
    VARRecord("P. Melin", 7, 12, 110, 1.10, 0.18, 0.07),
    VARRecord("S. Attwell", 19, 68, 410, 2.05, 0.43, 0.17),
]


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class RefVARAnalyzer:
    """
    Builds referee–VAR pair profiles and predicts card / penalty markets.

    Parameters
    ----------
    referees : optional override list of RefereeRecord
    var_officials : optional override list of VARRecord
    min_edge_pct : minimum edge % to flag as value in find_edges()
    """

    def __init__(
        self,
        referees: Optional[List[RefereeRecord]] = None,
        var_officials: Optional[List[VARRecord]] = None,
        min_edge_pct: float = 5.0,
    ) -> None:
        self._refs: Dict[str, RefereeRecord] = {
            r.name.lower(): r for r in (referees or _REFEREE_DB)
        }
        self._vars: Dict[str, VARRecord] = {
            v.name.lower(): v for v in (var_officials or _VAR_DB)
        }
        self._min_edge = min_edge_pct

    def analyse_pairing(
        self,
        referee_name: str,
        var_name: str,
        home_team: str = "",
        away_team: str = "",
        home_aggression: float = 1.0,
        away_aggression: float = 1.0,
    ) -> PairingAnalysis:
        ref = self._get_referee(referee_name)
        var_off = self._get_var(var_name)

        if ref is None:
            logger.warning("Referee '%s' not found — using defaults.", referee_name)
            ref = RefereeRecord(referee_name, 15, 40, 280, 3.80, 0.09, 0.28, 22.0, 0.22)
        if var_off is None:
            logger.warning("VAR '%s' not found — using defaults.", var_name)
            var_off = VARRecord(var_name, 12, 30, 200, 1.60, 0.30, 0.12)

        pairing = RefVARPairing(
            referee=ref,
            var_official=var_off,
            home_team=home_team,
            away_team=away_team,
            home_aggression=home_aggression,
            away_aggression=away_aggression,
        )
        return self._compute_analysis(pairing)

    def find_edges(
        self, analysis: PairingAnalysis, market_odds: Dict[str, float]
    ) -> List[PairingEdge]:
        """Compare model probabilities to bookmaker odds."""
        mapping = {
            "penalty_yes": analysis.p_penalty_in_match,
            "over_45y": analysis.p_over_45_yellows,
            "over_35y": analysis.p_over_35_yellows,
            "red_yes": analysis.p_over_05_reds,
        }
        edges: List[PairingEdge] = []
        for market, prob in mapping.items():
            if market not in market_odds:
                continue
            odds = market_odds[market]
            implied = 1.0 / odds if odds > 1.0 else 1.0
            edge_pct = (prob - implied) / implied * 100.0
            edges.append(
                PairingEdge(
                    market=market,
                    model_prob=prob,
                    book_odds=odds,
                    book_implied=implied,
                    edge_pct=edge_pct,
                    is_value=edge_pct >= self._min_edge,
                )
            )
        return sorted(edges, key=lambda e: e.edge_pct, reverse=True)

    def authority_network(self) -> List[Tuple[str, str, float]]:
        """All known ref–VAR combinations sorted by authority gap (largest first)."""
        pairs: List[Tuple[str, str, float]] = []
        for ref in self._refs.values():
            for var_off in self._vars.values():
                gap = ref.authority_score - var_off.authority_score
                pairs.append((ref.name, var_off.name, gap))
        return sorted(pairs, key=lambda t: t[2], reverse=True)

    def pairing_history(self, referee_name: str, var_name: str) -> Dict[str, float]:
        """Stub: returns per-pair historical stats (authority gap + expected OFR rate)."""
        ref = self._get_referee(referee_name)
        var_off = self._get_var(var_name)
        if ref and var_off:
            gap = ref.authority_score - var_off.authority_score
            ofr_mult = 1.0 / (1.0 + max(gap, 0) * 0.08)
            return {"authority_gap": gap, "ofr_rate_multiplier": ofr_mult}
        return {}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_analysis(self, pairing: RefVARPairing) -> PairingAnalysis:
        ref = pairing.referee
        var_off = pairing.var_official

        gap = ref.authority_score - var_off.authority_score
        ofr_mult = 1.0 / (1.0 + max(gap, 0.0) * 0.08)

        combined_aggression = (pairing.home_aggression + pairing.away_aggression) / 2.0

        # Expected cards (ref base × aggression × OFR effect on strict calls)
        exp_yellows = ref.avg_yellows_per_match * combined_aggression
        exp_reds = ref.avg_reds_per_match * combined_aggression

        # Expected penalties: base × OFR multiplier (fewer OFRs = fewer overturns = fewer awards)
        exp_penalties = self._compute_expected_penalty(pairing, ofr_mult)

        # Poisson probabilities
        p_penalty = self._poisson_gt(0.5, exp_penalties)
        p_over45y = self._poisson_gt(4.5, exp_yellows)
        p_over35y = self._poisson_gt(3.5, exp_yellows)
        p_red = self._poisson_gt(0.5, exp_reds)

        notes: List[str] = []
        if gap > 5.0:
            notes.append(
                f"Large authority gap ({gap:.1f}) — junior VAR unlikely to contradict senior ref"
            )
        if ofr_mult < 0.80:
            notes.append(
                f"OFR rate suppressed (×{ofr_mult:.2f}) — Under Penalties edge"
            )
        if combined_aggression > 1.20:
            notes.append(
                f"High combined aggression ({combined_aggression:.2f}) — cards market inflated"
            )

        # Market recommendations
        recommended: List[str] = []
        if p_penalty < 0.35:
            recommended.append("UNDER Penalties / No Penalty")
        if p_over45y < 0.40:
            recommended.append("UNDER 4.5 Yellow Cards")
        if p_over45y > 0.55:
            recommended.append("OVER 4.5 Yellow Cards")
        if p_red > 0.20:
            recommended.append("Red Card YES")

        return PairingAnalysis(
            pairing=pairing,
            authority_gap=gap,
            ofr_multiplier=ofr_mult,
            expected_penalties=exp_penalties,
            expected_yellows=exp_yellows,
            expected_reds=exp_reds,
            p_penalty_in_match=p_penalty,
            p_over_45_yellows=p_over45y,
            p_over_35_yellows=p_over35y,
            p_over_05_reds=p_red,
            recommended_markets=recommended,
            notes=notes,
        )

    def _compute_expected_penalty(
        self, pairing: RefVARPairing, ofr_mult: float
    ) -> float:
        ref = pairing.referee
        combined_aggression = (pairing.home_aggression + pairing.away_aggression) / 2.0
        base = ref.avg_penalties_awarded * combined_aggression
        # OFR reduces penalty rate (fewer monitor reviews = fewer soft-penalty overturns
        # AND fewer wrong non-calls corrected)
        return base * ofr_mult

    def _get_referee(self, name: str) -> Optional[RefereeRecord]:
        return self._refs.get(name.lower())

    def _get_var(self, name: str) -> Optional[VARRecord]:
        return self._vars.get(name.lower())

    @staticmethod
    def _poisson_cdf(k_max: int, lam: float) -> float:
        if k_max < 0:
            return 0.0
        if lam <= 0.0:
            return 1.0
        log_lam = math.log(lam)
        total = 0.0
        log_term = -lam
        for i in range(k_max + 1):
            if i > 0:
                log_term += log_lam - math.log(i)
            total += math.exp(log_term)
        return min(total, 1.0)

    def _poisson_gt(self, threshold: float, lam: float) -> float:
        return max(0.0, 1.0 - self._poisson_cdf(int(math.floor(threshold)), lam))


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------


def quick_pairing_check(
    referee_name: str,
    var_name: str,
    home_aggression: float = 1.0,
    away_aggression: float = 1.0,
) -> PairingAnalysis:
    """Instantiate analyzer with defaults and return pairing analysis."""
    return RefVARAnalyzer().analyse_pairing(
        referee_name=referee_name,
        var_name=var_name,
        home_aggression=home_aggression,
        away_aggression=away_aggression,
    )

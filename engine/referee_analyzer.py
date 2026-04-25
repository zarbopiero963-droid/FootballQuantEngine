"""
Referee Analyzer — Card Market Prediction Module
=================================================

Methodology
-----------
This module predicts football card market outcomes by combining referee historical
card rates with team foul tendencies.

  expected_yellows = ref_avg_yellows
                     + 0.5 × (home_foul_tendency + away_foul_tendency
                               - league_avg_fouls × 2)
                       × ref_card_rate_per_foul

  expected_reds    = ref_avg_reds   (Poisson, rarely adjusted due to low counts)

Probabilities are derived from the Poisson distribution, computed from scratch
(no scipy dependency):

  P(X > k) = 1 - sum_{i=0}^{floor(k)} e^{-λ} × λ^i / i!

Markets Targeted
----------------
- Over 3.5 yellows  (P_over_35_yellows)
- Over 4.5 yellows  (P_over_45_yellows)  ← primary yellow market
- Over 5.5 yellows  (P_over_55_yellows)
- Red card: YES     (P_red_card)          ← P(at least 1 red card)
- Over 4.5 total cards (P_over_45_total_cards)

Usage Example
-------------
    from engine.referee_analyzer import from_dicts

    rows = [
        {
            "referee": "M. Oliver",
            "home_team": "Arsenal",
            "away_team": "Chelsea",
            "home_yellows": 2,
            "away_yellows": 3,
            "home_reds": 0,
            "away_reds": 1,
            "home_fouls": 11,
            "away_fouls": 14,
        },
        # ... more historical rows
    ]

    analyzer = from_dicts(rows)
    prediction = analyzer.predict("M. Oliver", "Arsenal", "Chelsea")
    print(prediction)

    value_bets = analyzer.find_value_markets(
        prediction,
        market_odds_over45y=1.90,
        market_odds_red_yes=3.20,
    )
    for bet in value_bets:
        print(bet)
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level helpers (no scipy)
# ---------------------------------------------------------------------------


def _poisson_cdf(k_max: int, lam: float) -> float:
    """
    P(X <= k_max) for Poisson(lambda=lam).

    Uses the series definition:  sum_{i=0}^{k_max} e^{-lam} * lam^i / i!

    Edge cases:
    - lam <= 0  → P(X <= k_max) = 1.0 for k_max >= 0, else 0.0
    - k_max < 0 → 0.0
    """
    if k_max < 0:
        return 0.0
    if lam <= 0.0:
        return 1.0

    # Compute in log space to avoid overflow for large lam / k_max
    log_lam = math.log(lam)
    log_exp_neg_lam = -lam  # log(e^{-lam})

    total = 0.0
    log_term = log_exp_neg_lam  # i=0: log(e^{-lam} * lam^0 / 0!) = -lam
    for i in range(k_max + 1):
        if i > 0:
            log_term += log_lam - math.log(i)
        total += math.exp(log_term)

    return min(total, 1.0)


def _poisson_gt(threshold: float, lam: float) -> float:
    """
    P(X > threshold) = 1 - P(X <= floor(threshold))  for Poisson(lambda=lam).

    threshold may be fractional (e.g., 4.5 → floor = 4).
    """
    k_max = int(math.floor(threshold))
    return max(0.0, 1.0 - _poisson_cdf(k_max, lam))


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class MatchRefData:
    """One match record for referee analysis."""

    referee: str
    home_team: str
    away_team: str
    home_yellows: int
    away_yellows: int
    home_reds: int
    away_reds: int
    home_fouls: Optional[int] = None
    away_fouls: Optional[int] = None

    @property
    def total_yellows(self) -> int:
        return self.home_yellows + self.away_yellows

    @property
    def total_reds(self) -> int:
        return self.home_reds + self.away_reds

    @property
    def total_cards(self) -> int:
        return self.total_yellows + self.total_reds


@dataclass
class RefereeProfile:
    """Aggregated statistics for one referee."""

    referee: str
    matches: int
    avg_yellows: float  # per match
    avg_reds: float  # per match
    avg_total_cards: float  # per match
    avg_fouls: float  # total fouls per match (0 if not available)
    card_rate_per_foul: float  # avg_yellows / avg_fouls if avg_fouls > 0 else 0
    strictness_score: float  # normalised (0–1): avg_total_cards / 10, clipped


@dataclass
class TeamFoulProfile:
    """Aggregated foul and card statistics for one team."""

    team: str
    matches: int
    avg_fouls_committed: float  # per match
    avg_fouls_suffered: float  # per match
    avg_yellows_received: float  # per match


@dataclass
class CardMarketPrediction:
    """Full card-market prediction for one referee × match combination."""

    referee: str
    home_team: str
    away_team: str
    expected_total_yellows: float
    expected_total_reds: float
    expected_total_cards: float
    p_over_35_yellows: float  # P(total yellows > 3.5)
    p_over_45_yellows: float  # P(total yellows > 4.5)
    p_over_55_yellows: float  # P(total yellows > 5.5)
    p_red_card: float  # P(at least 1 red card)
    p_over_45_total_cards: float
    notes: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [
            "=" * 56,
            "Card Market Prediction",
            f"  Referee  : {self.referee}",
            f"  Match    : {self.home_team} vs {self.away_team}",
            "=" * 56,
            f"  Expected yellows   : {self.expected_total_yellows:.2f}",
            f"  Expected reds      : {self.expected_total_reds:.2f}",
            f"  Expected total     : {self.expected_total_cards:.2f}",
            "-" * 56,
            f"  P(yellows > 3.5)   : {self.p_over_35_yellows:.3f}  ({self.p_over_35_yellows * 100:.1f}%)",
            f"  P(yellows > 4.5)   : {self.p_over_45_yellows:.3f}  ({self.p_over_45_yellows * 100:.1f}%)",
            f"  P(yellows > 5.5)   : {self.p_over_55_yellows:.3f}  ({self.p_over_55_yellows * 100:.1f}%)",
            f"  P(red card YES)    : {self.p_red_card:.3f}  ({self.p_red_card * 100:.1f}%)",
            f"  P(total > 4.5)     : {self.p_over_45_total_cards:.3f}  ({self.p_over_45_total_cards * 100:.1f}%)",
        ]
        if self.notes:
            lines.append("-" * 56)
            for note in self.notes:
                lines.append(f"  * {note}")
        lines.append("=" * 56)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main analyzer
# ---------------------------------------------------------------------------


class RefereeAnalyzer:
    """
    Builds referee and team profiles from historical match data.

    The card-count model uses a Poisson approximation:
      P(cards > threshold) = 1 - CDF(floor(threshold), lambda=expected)

    Expected yellows for a match:
      base       = referee_avg_yellows
      adjustment = 0.5 × (home_foul_tendency + away_foul_tendency
                          - league_avg_fouls × 2)
                   × ref_card_rate_per_foul
      expected_yellows = max(0.5, base + adjustment)

    Expected reds: independent Poisson, lambda = ref_avg_reds
    (rarely adjusted since counts are low and noisy).

    League-average fallbacks are used when a referee or team is not found in
    the fitted data.
    """

    # Sensible league-wide fallback constants
    _FALLBACK_AVG_YELLOWS: float = 3.8
    _FALLBACK_AVG_REDS: float = 0.18
    _FALLBACK_AVG_FOULS: float = 22.0
    _FALLBACK_CARD_RATE_PER_FOUL: float = 3.8 / 22.0

    def __init__(self) -> None:
        self._ref_profiles: Dict[str, RefereeProfile] = {}
        self._team_profiles: Dict[str, TeamFoulProfile] = {}
        self._league_avg_fouls: float = 22.0  # recalculated on fit()

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, matches: List[MatchRefData]) -> "RefereeAnalyzer":
        """
        Build referee and team profiles from historical match data.
        Recalculates league-wide average fouls from the data when available.
        Returns self for chaining.
        """
        if not matches:
            logger.warning("RefereeAnalyzer.fit() called with empty match list.")
            return self

        self._ref_profiles = self._build_referee_profiles(matches)
        self._team_profiles = self._build_team_profiles(matches)
        self._league_avg_fouls = self._compute_league_avg_fouls(matches)

        logger.info(
            "RefereeAnalyzer fitted: %d referees, %d teams, "
            "league_avg_fouls=%.2f from %d matches.",
            len(self._ref_profiles),
            len(self._team_profiles),
            self._league_avg_fouls,
            len(matches),
        )
        return self

    def _build_referee_profiles(
        self, matches: List[MatchRefData]
    ) -> Dict[str, RefereeProfile]:
        """Aggregate per-referee statistics."""
        # Accumulators: ref -> list of per-match values
        acc: Dict[str, Dict[str, list]] = defaultdict(
            lambda: {
                "yellows": [],
                "reds": [],
                "cards": [],
                "fouls": [],
            }
        )

        for m in matches:
            acc[m.referee]["yellows"].append(m.total_yellows)
            acc[m.referee]["reds"].append(m.total_reds)
            acc[m.referee]["cards"].append(m.total_cards)
            if m.home_fouls is not None and m.away_fouls is not None:
                total_fouls = m.home_fouls + m.away_fouls
                if total_fouls > 0:
                    acc[m.referee]["fouls"].append(total_fouls)

        profiles: Dict[str, RefereeProfile] = {}
        for ref, data in acc.items():
            n = len(data["yellows"])
            avg_y = sum(data["yellows"]) / n
            avg_r = sum(data["reds"]) / n
            avg_c = sum(data["cards"]) / n

            if data["fouls"]:
                avg_f = sum(data["fouls"]) / len(data["fouls"])
                rate = avg_y / avg_f if avg_f > 0 else 0.0
            else:
                avg_f = 0.0
                rate = 0.0

            strictness = min(avg_c / 10.0, 1.0)

            profiles[ref] = RefereeProfile(
                referee=ref,
                matches=n,
                avg_yellows=avg_y,
                avg_reds=avg_r,
                avg_total_cards=avg_c,
                avg_fouls=avg_f,
                card_rate_per_foul=rate,
                strictness_score=strictness,
            )
        return profiles

    def _build_team_profiles(
        self, matches: List[MatchRefData]
    ) -> Dict[str, TeamFoulProfile]:
        """Aggregate per-team foul and yellow card statistics."""
        # team -> {committed, suffered, yellows_received, n}
        acc: Dict[str, Dict[str, list]] = defaultdict(
            lambda: {
                "committed": [],
                "suffered": [],
                "yellows": [],
            }
        )

        for m in matches:
            if m.home_fouls is not None:
                acc[m.home_team]["committed"].append(m.home_fouls)
                acc[m.away_team]["suffered"].append(m.home_fouls)
            if m.away_fouls is not None:
                acc[m.home_team]["suffered"].append(m.away_fouls)
                acc[m.away_team]["committed"].append(m.away_fouls)
            acc[m.home_team]["yellows"].append(m.home_yellows)
            acc[m.away_team]["yellows"].append(m.away_yellows)

        profiles: Dict[str, TeamFoulProfile] = {}
        for team, data in acc.items():
            n = len(data["yellows"])
            profiles[team] = TeamFoulProfile(
                team=team,
                matches=n,
                avg_fouls_committed=sum(data["committed"]) / n,
                avg_fouls_suffered=sum(data["suffered"]) / n,
                avg_yellows_received=sum(data["yellows"]) / n,
            )
        return profiles

    def _compute_league_avg_fouls(self, matches: List[MatchRefData]) -> float:
        """League-wide average total fouls per match (only from matches that have foul data)."""
        fouls_per_match = [
            m.home_fouls + m.away_fouls
            for m in matches
            if m.home_fouls is not None and m.away_fouls is not None
        ]
        if fouls_per_match:
            return sum(fouls_per_match) / len(fouls_per_match)
        return self._FALLBACK_AVG_FOULS

    # ------------------------------------------------------------------
    # Profile accessors
    # ------------------------------------------------------------------

    def get_referee_profile(self, referee: str) -> Optional[RefereeProfile]:
        """Return the fitted RefereeProfile, or None if not found."""
        return self._ref_profiles.get(referee)

    def get_team_profile(self, team: str) -> Optional[TeamFoulProfile]:
        """Return the fitted TeamFoulProfile, or None if not found."""
        return self._team_profiles.get(team)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        referee: str,
        home_team: str,
        away_team: str,
    ) -> CardMarketPrediction:
        """
        Predict card market outcomes for a referee × match combination.

        Falls back to league-wide averages when profiles are unavailable.
        Poisson CDF is used for all probability calculations.
        """
        notes: List[str] = []

        # ---- Referee profile -------------------------------------------
        ref_profile = self._ref_profiles.get(referee)
        if ref_profile is None:
            logger.debug("Referee '%s' not in history; using fallbacks.", referee)
            notes.append(
                f"Referee '{referee}' not in history — using league fallbacks."
            )
            ref_avg_y = self._FALLBACK_AVG_YELLOWS
            ref_avg_r = self._FALLBACK_AVG_REDS
            ref_card_rate = self._FALLBACK_CARD_RATE_PER_FOUL
        else:
            ref_avg_y = ref_profile.avg_yellows
            ref_avg_r = ref_profile.avg_reds
            # Use card_rate_per_foul from profile; fall back if no foul data
            ref_card_rate = (
                ref_profile.card_rate_per_foul
                if ref_profile.avg_fouls > 0
                else self._FALLBACK_CARD_RATE_PER_FOUL
            )
            if ref_profile.strictness_score >= 0.6:
                notes.append(
                    f"Strict referee: avg {ref_profile.avg_total_cards:.1f} cards/match "
                    f"(strictness={ref_profile.strictness_score:.2f})"
                )
            if ref_profile.matches < 10:
                notes.append(
                    f"Small sample for referee '{referee}': only {ref_profile.matches} matches."
                )

        # ---- Team foul tendencies --------------------------------------
        home_prof = self._team_profiles.get(home_team)
        away_prof = self._team_profiles.get(away_team)

        league_avg = self._league_avg_fouls  # total per match

        if home_prof is None:
            logger.debug(
                "Home team '%s' not in history; using league average.", home_team
            )
            notes.append(
                f"Home team '{home_team}' not in history — using league avg fouls."
            )
            home_foul_tendency = league_avg / 2.0
        else:
            home_foul_tendency = home_prof.avg_fouls_committed
            if home_prof.matches < 5:
                notes.append(
                    f"Small sample for '{home_team}': {home_prof.matches} matches."
                )

        if away_prof is None:
            logger.debug(
                "Away team '%s' not in history; using league average.", away_team
            )
            notes.append(
                f"Away team '{away_team}' not in history — using league avg fouls."
            )
            away_foul_tendency = league_avg / 2.0
        else:
            away_foul_tendency = away_prof.avg_fouls_committed
            if away_prof.matches < 5:
                notes.append(
                    f"Small sample for '{away_team}': {away_prof.matches} matches."
                )

        # ---- Expected yellows ------------------------------------------
        # adjustment = 0.5 × (home_foul_tendency + away_foul_tendency - league_avg×2)
        #              × ref_card_rate_per_foul
        foul_deviation = (home_foul_tendency + away_foul_tendency) - (league_avg)
        # league_avg already represents total (home+away) per match
        adjustment = 0.5 * foul_deviation * ref_card_rate
        expected_yellows = max(0.5, ref_avg_y + adjustment)

        # ---- Expected reds ---------------------------------------------
        expected_reds = max(0.0, ref_avg_r)

        # ---- Expected total cards -------------------------------------
        expected_total = expected_yellows + expected_reds

        # ---- Poisson probabilities ------------------------------------
        p_over_35y = _poisson_gt(3.5, expected_yellows)
        p_over_45y = _poisson_gt(4.5, expected_yellows)
        p_over_55y = _poisson_gt(5.5, expected_yellows)

        # P(red card YES) = P(reds >= 1) = 1 - P(reds == 0)
        # For Poisson: P(X=0) = e^{-lambda}
        p_red = 1.0 - math.exp(-expected_reds) if expected_reds > 0 else 0.0

        # Over 4.5 total cards: sum of yellows + reds as compound Poisson
        # Approximation: treat total cards as Poisson(lambda = expected_total)
        p_over_45_total = _poisson_gt(4.5, expected_total)

        return CardMarketPrediction(
            referee=referee,
            home_team=home_team,
            away_team=away_team,
            expected_total_yellows=round(expected_yellows, 3),
            expected_total_reds=round(expected_reds, 3),
            expected_total_cards=round(expected_total, 3),
            p_over_35_yellows=round(p_over_35y, 4),
            p_over_45_yellows=round(p_over_45y, 4),
            p_over_55_yellows=round(p_over_55y, 4),
            p_red_card=round(p_red, 4),
            p_over_45_total_cards=round(p_over_45_total, 4),
            notes=notes,
        )

    # ------------------------------------------------------------------
    # Value market finder
    # ------------------------------------------------------------------

    def find_value_markets(
        self,
        prediction: CardMarketPrediction,
        market_odds_over35y: Optional[float] = None,
        market_odds_over45y: Optional[float] = None,
        market_odds_red_yes: Optional[float] = None,
        min_edge: float = 0.04,
    ) -> List[Dict]:
        """
        Compare model probabilities against decimal market odds.

        For each market where odds are supplied:
          implied_prob = 1 / odds
          edge = model_prob - implied_prob

        Returns list of dicts (sorted by edge descending) with:
          {"market", "model_prob", "implied", "edge", "odds"}

        Only markets with edge >= min_edge are included.
        """
        candidates: List[Tuple[str, float, Optional[float]]] = [
            ("Over 3.5 Yellows", prediction.p_over_35_yellows, market_odds_over35y),
            ("Over 4.5 Yellows", prediction.p_over_45_yellows, market_odds_over45y),
            ("Red Card YES", prediction.p_red_card, market_odds_red_yes),
        ]

        results: List[Dict] = []
        for market_name, model_prob, odds in candidates:
            if odds is None:
                continue
            if odds <= 1.0:
                logger.warning(
                    "Invalid decimal odds %.4f for market '%s'; skipping.",
                    odds,
                    market_name,
                )
                continue
            implied = 1.0 / odds
            edge = model_prob - implied
            if edge >= min_edge:
                results.append(
                    {
                        "market": market_name,
                        "model_prob": round(model_prob, 4),
                        "implied": round(implied, 4),
                        "edge": round(edge, 4),
                        "odds": odds,
                    }
                )

        results.sort(key=lambda x: x["edge"], reverse=True)
        return results


# ---------------------------------------------------------------------------
# Module-level convenience constructor
# ---------------------------------------------------------------------------


def from_dicts(rows: List[dict]) -> "RefereeAnalyzer":
    """
    Convenience function: fit a RefereeAnalyzer from a list of plain dicts.

    Required keys per dict:
        referee, home_team, away_team,
        home_yellows, away_yellows, home_reds, away_reds

    Optional keys:
        home_fouls, away_fouls  (omit or None if not available)

    Returns a fitted RefereeAnalyzer instance.
    """
    matches: List[MatchRefData] = []
    for i, row in enumerate(rows):
        try:
            m = MatchRefData(
                referee=str(row["referee"]),
                home_team=str(row["home_team"]),
                away_team=str(row["away_team"]),
                home_yellows=int(row["home_yellows"]),
                away_yellows=int(row["away_yellows"]),
                home_reds=int(row["home_reds"]),
                away_reds=int(row["away_reds"]),
                home_fouls=int(row["home_fouls"])
                if row.get("home_fouls") is not None
                else None,
                away_fouls=int(row["away_fouls"])
                if row.get("away_fouls") is not None
                else None,
            )
            matches.append(m)
        except (KeyError, ValueError, TypeError) as exc:
            logger.warning("Skipping row %d due to error: %s | row=%s", i, exc, row)

    analyzer = RefereeAnalyzer()
    analyzer.fit(matches)
    return analyzer

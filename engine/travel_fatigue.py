"""
Travel & Circadian Rhythm Fatigue Engine
=========================================

Models the performance penalty caused by long-haul travel, compressed fixture
schedules, and circadian-rhythm disruption (jet lag) for football teams.

Particularly effective for:
- UEFA / CONMEBOL cup competitions (midweek + weekend congestion)
- South American qualifiers (altitude + timezone + distance)
- Teams travelling across multiple time zones in 48-72 hours

Fatigue components
------------------
1. Travel distance   km / 1000 × 0.08           (8% per 1000 km)
2. Rest days         max(0, (5 - days_rest) × 0.12)
3. Jet lag           abs(tz_diff_hours) × 0.04   (away team only)
4. Altitude          max(0, (alt_m - 500) / 1000 × 0.15)
5. Congestion        max(0, (games_in_10_days - 2) × 0.08)

Combined: min(total, 0.60)
λ multiplier: max(0.65, 1.0 - total_fatigue × 0.45)

Usage
-----
    from engine.travel_fatigue import TravelFatigueEngine

    engine = TravelFatigueEngine()
    report = engine.compare_teams(
        home_team="Arsenal",
        away_team="Bayern Munich",
        venue="emirates",
        home_days_rest=6,
        away_days_rest=3,
    )
    print(report)
    lh, la = engine.adjust_lambdas(1.55, 1.20, report)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stadium registry
# ---------------------------------------------------------------------------

# Each entry: lat, lon, alt_m (metres above sea level), tz (UTC offset hours)
STADIUM_COORDS: Dict[str, Dict[str, float]] = {
    # England
    "old_trafford": {"lat": 53.463, "lon": -2.291, "alt_m": 43, "tz": 0},
    "anfield": {"lat": 53.431, "lon": -2.961, "alt_m": 16, "tz": 0},
    "stamford_bridge": {"lat": 51.482, "lon": -0.191, "alt_m": 14, "tz": 0},
    "emirates": {"lat": 51.555, "lon": -0.108, "alt_m": 38, "tz": 0},
    "etihad": {"lat": 53.483, "lon": -2.200, "alt_m": 46, "tz": 0},
    "tottenham_hotspur": {"lat": 51.604, "lon": -0.066, "alt_m": 30, "tz": 0},
    "villa_park": {"lat": 52.509, "lon": -1.885, "alt_m": 97, "tz": 0},
    "st_james_park": {"lat": 54.975, "lon": -1.622, "alt_m": 30, "tz": 0},
    "goodison_park": {"lat": 53.439, "lon": -2.966, "alt_m": 21, "tz": 0},
    "elland_road": {"lat": 53.778, "lon": -1.572, "alt_m": 51, "tz": 0},
    # Spain
    "santiago_bernabeu": {"lat": 40.453, "lon": -3.688, "alt_m": 667, "tz": 1},
    "camp_nou": {"lat": 41.381, "lon": 2.123, "alt_m": 57, "tz": 1},
    "metropolitano": {"lat": 40.436, "lon": -3.600, "alt_m": 586, "tz": 1},
    "mestalla": {"lat": 39.475, "lon": -0.359, "alt_m": 15, "tz": 1},
    "san_mames": {"lat": 43.264, "lon": -2.950, "alt_m": 22, "tz": 1},
    "estadio_benito_villamarin": {"lat": 37.356, "lon": -5.981, "alt_m": 12, "tz": 1},
    # Italy
    "san_siro": {"lat": 45.478, "lon": 9.124, "alt_m": 122, "tz": 1},
    "olimpico_roma": {"lat": 41.934, "lon": 12.455, "alt_m": 20, "tz": 1},
    "juventus_stadium": {"lat": 45.110, "lon": 7.641, "alt_m": 237, "tz": 1},
    "san_paolo": {"lat": 40.828, "lon": 14.193, "alt_m": 9, "tz": 1},
    "gewiss_stadium": {"lat": 45.709, "lon": 9.680, "alt_m": 212, "tz": 1},
    # Germany
    "allianz_arena": {"lat": 48.219, "lon": 11.625, "alt_m": 520, "tz": 1},
    "signal_iduna_park": {"lat": 51.493, "lon": 7.452, "alt_m": 86, "tz": 1},
    "red_bull_arena": {"lat": 51.346, "lon": 12.348, "alt_m": 115, "tz": 1},
    "bayarena": {"lat": 51.038, "lon": 7.002, "alt_m": 68, "tz": 1},
    "volksparkstadion": {"lat": 53.587, "lon": 10.006, "alt_m": 7, "tz": 1},
    # France
    "parc_des_princes": {"lat": 48.841, "lon": 2.253, "alt_m": 35, "tz": 1},
    "velodrome": {"lat": 43.270, "lon": 5.396, "alt_m": 5, "tz": 1},
    "groupama_stadium": {"lat": 45.765, "lon": 4.982, "alt_m": 200, "tz": 1},
    # Netherlands
    "johan_cruyff_arena": {"lat": 52.314, "lon": 4.942, "alt_m": -2, "tz": 1},
    "de_kuip": {"lat": 51.894, "lon": 4.523, "alt_m": 0, "tz": 1},
    # Portugal
    "estadio_da_luz": {"lat": 38.753, "lon": -9.184, "alt_m": 100, "tz": 0},
    "estadio_dragao": {"lat": 41.162, "lon": -8.583, "alt_m": 118, "tz": 0},
    # Turkey
    "ataturk_olympic": {"lat": 41.074, "lon": 28.766, "alt_m": 50, "tz": 3},
    "sukru_saracoglu": {"lat": 41.016, "lon": 29.032, "alt_m": 30, "tz": 3},
    # Russia
    "luzhniki": {"lat": 55.716, "lon": 37.556, "alt_m": 145, "tz": 3},
    # High-altitude stadiums
    "estadio_monumental": {"lat": -34.545, "lon": -58.450, "alt_m": 8, "tz": -3},
    "estadio_mas_monumental": {"lat": -34.545, "lon": -58.450, "alt_m": 8, "tz": -3},
    "estadio_hernando_siles": {"lat": -16.508, "lon": -68.120, "alt_m": 3637, "tz": -4},
    "estadio_atahualpa": {"lat": -0.183, "lon": -78.484, "alt_m": 2850, "tz": -5},
    "estadio_mineirao": {"lat": -19.865, "lon": -43.970, "alt_m": 858, "tz": -3},
    "estadio_azteca": {"lat": 19.303, "lon": -99.151, "alt_m": 2240, "tz": -6},
    # Israel
    "bloomfield": {"lat": 32.047, "lon": 34.762, "alt_m": 30, "tz": 2},
    # Scotland
    "celtic_park": {"lat": 55.850, "lon": -4.205, "alt_m": 46, "tz": 0},
    "ibrox": {"lat": 55.851, "lon": -4.309, "alt_m": 20, "tz": 0},
}

_OPTIMAL_REST_DAYS = 5
_MAX_FATIGUE = 0.60
_LAMBDA_FLOOR = 0.65


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometres using the Haversine formula."""
    r = 6371.0
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    lat1_r = math.radians(lat1)
    lat2_r = math.radians(lat2)
    a = (
        math.sin(d_lat / 2) ** 2
        + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(d_lon / 2) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return r * c


def _slugify(name: str) -> str:
    result = name.lower()
    result = result.replace(" ", "_").replace("-", "_").replace("'", "")
    return result


def _count_games_in_window(
    fixtures: List[Dict], reference_date: str, window_days: int = 10
) -> int:
    """
    Count fixtures within *window_days* before *reference_date*.

    Dates must be ISO format strings "YYYY-MM-DD".
    Returns 0 if date parsing fails.
    """
    try:
        ref_parts = [int(p) for p in reference_date.split("-")]
        ref_days = ref_parts[0] * 365 + ref_parts[1] * 30 + ref_parts[2]
    except Exception:
        return 0

    count = 0
    for fx in fixtures:
        try:
            parts = [int(p) for p in fx["date"].split("-")]
            fx_days = parts[0] * 365 + parts[1] * 30 + parts[2]
            age = ref_days - fx_days
            if 0 < age <= window_days:
                count += 1
        except Exception:
            continue
    return count


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class TravelAssessment:
    """Full fatigue breakdown for one team ahead of a fixture."""

    team: str
    venue_slug: str
    distance_km: float
    days_rest: int
    timezone_diff_hours: float
    altitude_m: float
    games_in_last_10_days: int
    fatigue_travel: float
    fatigue_rest: float
    fatigue_jetlag: float
    fatigue_altitude: float
    fatigue_congestion: float
    total_fatigue: float
    lambda_multiplier: float
    warning: str

    def summary(self) -> str:
        lines = [
            f"Team          : {self.team}",
            f"Venue         : {self.venue_slug}",
            f"Distance      : {self.distance_km:.0f} km",
            f"Days rest     : {self.days_rest}",
            f"TZ diff       : {self.timezone_diff_hours:.1f} h",
            f"Altitude      : {self.altitude_m:.0f} m",
            f"Games / 10d   : {self.games_in_last_10_days}",
            "──────────────────────────────",
            f"  Travel fat. : {self.fatigue_travel:.3f}",
            f"  Rest fat.   : {self.fatigue_rest:.3f}",
            f"  Jet lag fat.: {self.fatigue_jetlag:.3f}",
            f"  Altitude f. : {self.fatigue_altitude:.3f}",
            f"  Congestion  : {self.fatigue_congestion:.3f}",
            f"  TOTAL       : {self.total_fatigue:.3f}",
            "──────────────────────────────",
            f"  λ multiplier: {self.lambda_multiplier:.3f}",
        ]
        if self.warning:
            lines.append(f"  ⚠ {self.warning}")
        return "\n".join(lines)


@dataclass
class FatigueReport:
    """Comparative fatigue report for both teams in a fixture."""

    home_team: str
    away_team: str
    home_assessment: TravelAssessment
    away_assessment: TravelAssessment
    net_advantage: float  # positive = home advantage
    recommended_action: str

    def __str__(self) -> str:
        lines = [
            "=" * 50,
            "TRAVEL FATIGUE REPORT",
            "=" * 50,
            "",
            f"[HOME] {self.home_team}",
            self.home_assessment.summary(),
            "",
            f"[AWAY] {self.away_team}",
            self.away_assessment.summary(),
            "",
            "─" * 50,
            f"Net advantage (home positive): {self.net_advantage:+.3f}",
            f"Action: {self.recommended_action}",
            "=" * 50,
        ]
        return "\n".join(lines)


@dataclass
class FixtureSchedule:
    """Recent fixture history for one team."""

    team: str
    home_ground: str
    fixtures: List[Dict] = field(default_factory=list)
    # Each dict: {"date": "2025-04-10", "venue_slug": "anfield", "is_home": bool}


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class TravelFatigueEngine:
    """
    Assesses travel and circadian fatigue for football teams.

    Parameters
    ----------
    home_venue_overrides : dict
        Maps team name to their home venue slug, e.g.
        ``{"Arsenal": "emirates", "Bayern Munich": "allianz_arena"}``.
    """

    # Well-known team → home venue mappings (fallback when not supplied)
    _KNOWN_HOME_GROUNDS: Dict[str, str] = {
        "manchester united": "old_trafford",
        "manchester city": "etihad",
        "liverpool": "anfield",
        "chelsea": "stamford_bridge",
        "arsenal": "emirates",
        "tottenham": "tottenham_hotspur",
        "tottenham hotspur": "tottenham_hotspur",
        "aston villa": "villa_park",
        "newcastle": "st_james_park",
        "newcastle united": "st_james_park",
        "real madrid": "santiago_bernabeu",
        "barcelona": "camp_nou",
        "atletico madrid": "metropolitano",
        "atletico de madrid": "metropolitano",
        "inter milan": "san_siro",
        "ac milan": "san_siro",
        "juventus": "juventus_stadium",
        "napoli": "san_paolo",
        "roma": "olimpico_roma",
        "as roma": "olimpico_roma",
        "lazio": "olimpico_roma",
        "bayern munich": "allianz_arena",
        "fc bayern": "allianz_arena",
        "borussia dortmund": "signal_iduna_park",
        "rb leipzig": "red_bull_arena",
        "bayer leverkusen": "bayarena",
        "paris saint-germain": "parc_des_princes",
        "psg": "parc_des_princes",
        "olympique marseille": "velodrome",
        "porto": "estadio_dragao",
        "benfica": "estadio_da_luz",
        "ajax": "johan_cruyff_arena",
        "celtic": "celtic_park",
        "rangers": "ibrox",
    }

    def __init__(
        self,
        home_venue_overrides: Optional[Dict[str, str]] = None,
    ) -> None:
        self._overrides: Dict[str, str] = {}
        if home_venue_overrides:
            for team, venue in home_venue_overrides.items():
                self._overrides[team.lower()] = venue

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def assess_team(
        self,
        team: str,
        playing_at: str,
        home_ground: str,
        days_rest: int,
        recent_fixtures: Optional[List[Dict]] = None,
        is_home_team: bool = False,
        match_date: Optional[str] = None,
    ) -> TravelAssessment:
        """
        Compute the full fatigue assessment for one team.

        Parameters
        ----------
        team         : team name (for display)
        playing_at   : slug of the venue where the match is played
        home_ground  : slug of the team's own home ground
        days_rest    : full days since last match (0 = played yesterday)
        recent_fixtures : list of {"date": "YYYY-MM-DD", "venue_slug": str, "is_home": bool}
        is_home_team : True when the team plays on their own ground (no jet-lag)
        match_date   : ISO date of this match (for congestion count)
        """
        venue_coords = self._venue_coords(playing_at)
        home_coords = self._venue_coords(home_ground)

        # Distance
        if venue_coords and home_coords:
            distance_km = _haversine_km(
                home_coords["lat"],
                home_coords["lon"],
                venue_coords["lat"],
                venue_coords["lon"],
            )
        else:
            distance_km = 0.0
            logger.warning(
                "Could not resolve coordinates for venue=%s or home=%s",
                playing_at,
                home_ground,
            )

        # Altitude at playing venue
        altitude_m = venue_coords["alt_m"] if venue_coords else 0.0

        # Timezone difference (only relevant for away team)
        if is_home_team or venue_coords is None or home_coords is None:
            tz_diff = 0.0
        else:
            tz_diff = abs(venue_coords.get("tz", 0.0) - home_coords.get("tz", 0.0))

        # Games in last 10 days
        if recent_fixtures and match_date:
            games_10d = _count_games_in_window(recent_fixtures, match_date, 10)
        else:
            games_10d = 0

        # Fatigue components
        f_travel = min(distance_km / 1000.0 * 0.08, 0.30)
        f_rest = max(0.0, (_OPTIMAL_REST_DAYS - days_rest) * 0.12)
        f_jetlag = tz_diff * 0.04
        f_altitude = max(0.0, (altitude_m - 500.0) / 1000.0 * 0.15)
        f_congestion = max(0.0, (games_10d - 2) * 0.08)

        total = min(
            f_travel + f_rest + f_jetlag + f_altitude + f_congestion, _MAX_FATIGUE
        )
        lam_mult = max(_LAMBDA_FLOOR, 1.0 - total * 0.45)

        # Human-readable warning
        warning = self._make_warning(
            distance_km, days_rest, tz_diff, altitude_m, games_10d, total
        )

        return TravelAssessment(
            team=team,
            venue_slug=playing_at,
            distance_km=distance_km,
            days_rest=days_rest,
            timezone_diff_hours=tz_diff,
            altitude_m=altitude_m,
            games_in_last_10_days=games_10d,
            fatigue_travel=f_travel,
            fatigue_rest=f_rest,
            fatigue_jetlag=f_jetlag,
            fatigue_altitude=f_altitude,
            fatigue_congestion=f_congestion,
            total_fatigue=total,
            lambda_multiplier=lam_mult,
            warning=warning,
        )

    def compare_teams(
        self,
        home_team: str,
        away_team: str,
        venue: str,
        home_days_rest: int,
        away_days_rest: int,
        home_ground: Optional[str] = None,
        away_ground: Optional[str] = None,
        home_recent: Optional[List[Dict]] = None,
        away_recent: Optional[List[Dict]] = None,
        match_date: Optional[str] = None,
    ) -> FatigueReport:
        """
        Build a comparative FatigueReport for both teams.

        Home ground slugs are resolved automatically from the known-grounds
        table if not supplied.
        """
        home_slug = home_ground or self._resolve_home_ground(home_team) or venue
        away_slug = away_ground or self._resolve_home_ground(away_team) or venue

        home_assessment = self.assess_team(
            team=home_team,
            playing_at=venue,
            home_ground=home_slug,
            days_rest=home_days_rest,
            recent_fixtures=home_recent,
            is_home_team=True,
            match_date=match_date,
        )
        away_assessment = self.assess_team(
            team=away_team,
            playing_at=venue,
            home_ground=away_slug,
            days_rest=away_days_rest,
            recent_fixtures=away_recent,
            is_home_team=False,
            match_date=match_date,
        )

        net_adv = away_assessment.total_fatigue - home_assessment.total_fatigue
        action = self._recommend(net_adv, home_team, away_team)

        logger.info(
            "FatigueReport: home=%s (fat=%.2f lam=%.3f) away=%s (fat=%.2f lam=%.3f) net=%.3f",
            home_team,
            home_assessment.total_fatigue,
            home_assessment.lambda_multiplier,
            away_team,
            away_assessment.total_fatigue,
            away_assessment.lambda_multiplier,
            net_adv,
        )

        return FatigueReport(
            home_team=home_team,
            away_team=away_team,
            home_assessment=home_assessment,
            away_assessment=away_assessment,
            net_advantage=net_adv,
            recommended_action=action,
        )

    def adjust_lambdas(
        self,
        lambda_home: float,
        lambda_away: float,
        report: FatigueReport,
    ) -> Tuple[float, float]:
        """
        Apply fatigue multipliers to pre-match expected-goal rates.

        Returns
        -------
        (adjusted_lambda_home, adjusted_lambda_away)
        """
        adj_home = max(0.10, lambda_home * report.home_assessment.lambda_multiplier)
        adj_away = max(0.10, lambda_away * report.away_assessment.lambda_multiplier)
        logger.debug(
            "adjust_lambdas: lh %.3f→%.3f  la %.3f→%.3f",
            lambda_home,
            adj_home,
            lambda_away,
            adj_away,
        )
        return adj_home, adj_away

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _venue_coords(self, slug: str) -> Optional[Dict[str, float]]:
        key = _slugify(slug)
        return STADIUM_COORDS.get(key)

    def _resolve_home_ground(self, team_name: str) -> Optional[str]:
        key = team_name.lower()
        if key in self._overrides:
            return self._overrides[key]
        return self._KNOWN_HOME_GROUNDS.get(key)

    @staticmethod
    def _make_warning(
        km: float,
        rest: int,
        tz: float,
        alt: float,
        games: int,
        total: float,
    ) -> str:
        parts: List[str] = []
        if km > 3000:
            parts.append(f"long-haul travel ({km:.0f} km)")
        if rest <= 2:
            parts.append(f"only {rest}d rest")
        if tz >= 3:
            parts.append(f"{tz:.0f}h time-zone shift")
        if alt > 2000:
            parts.append(f"high altitude ({alt:.0f}m)")
        if games >= 4:
            parts.append(f"congested schedule ({games} games in 10d)")
        if not parts:
            if total < 0.10:
                return "No significant fatigue"
            return f"Mild fatigue (score={total:.2f})"
        return "; ".join(parts).capitalize() + f" — fatigue={total:.2f}"

    @staticmethod
    def _recommend(net_adv: float, home: str, away: str) -> str:
        if net_adv > 0.25:
            return f"BACK HOME ({home}) — away team heavily fatigued (Δ={net_adv:+.2f})"
        if net_adv > 0.10:
            return (
                f"LEAN HOME ({home}) — away team moderately fatigued (Δ={net_adv:+.2f})"
            )
        if net_adv < -0.25:
            return f"BACK AWAY ({away}) — home team surprisingly fatigued (Δ={net_adv:+.2f})"
        if net_adv < -0.10:
            return f"LEAN AWAY ({away}) — home team more tired (Δ={net_adv:+.2f})"
        return (
            f"No significant fatigue edge (Δ={net_adv:+.2f}) — other factors dominate"
        )


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------


def quick_fatigue_check(
    home_team_slug: str,
    away_team_slug: str,
    venue_slug: str,
    home_days_rest: int = 7,
    away_days_rest: int = 3,
    away_travel_km: float = 0.0,
) -> FatigueReport:
    """
    Fast-path wrapper for basic fatigue checks.

    If the venue slug is not in STADIUM_COORDS, supply ``away_travel_km``
    directly to override the distance calculation.
    """
    engine = TravelFatigueEngine()

    if away_travel_km > 0.0 and venue_slug not in STADIUM_COORDS:
        # Inject a synthetic venue entry so the engine can compute correctly
        STADIUM_COORDS[_slugify(venue_slug)] = {
            "lat": 0.0,
            "lon": 0.0,
            "alt_m": 0.0,
            "tz": 0.0,
        }

    return engine.compare_teams(
        home_team=home_team_slug,
        away_team=away_team_slug,
        venue=venue_slug,
        home_days_rest=home_days_rest,
        away_days_rest=away_days_rest,
    )

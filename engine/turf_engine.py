"""
Pitch & Turf Engine
===================

Models the physics of the playing surface and its effect on match outcomes.
Bookmakers price teams; this engine prices the *field*.

Key effects
-----------
- **Pitch dimensions**: narrow pitch (<66m wide) suppresses goals (Under edge);
  wide pitch (>70m) opens space and increases xG.
- **Wet grass** (irrigation + rain): ball speed +20% → favours possession/tiki-taka
  teams; increases total xG.
- **Grass length**: long grass (40+ mm) slows technical play, benefits direct/physical
  teams. Short grass (24mm, Etihad) amplifies Man City's passing game.
- **Synthetic turf**: higher bounce, faster pace; penalises technical dribblers,
  benefits direct/aerial teams.

Usage
-----
    from engine.turf_engine import TurfEngine

    engine = TurfEngine()
    impact = engine.analyse("etihad", "Manchester City", "Arsenal", rain_mm_per_hour=6.0)
    print(impact.summary())
    lh, la = engine.adjust_lambdas(1.80, 1.10, impact)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stadium turf database
# ---------------------------------------------------------------------------


@dataclass
class TurfProfile:
    """Static turf characteristics for one stadium."""

    stadium_slug: str
    length_m: float
    width_m: float
    turf_type: str  # "NATURAL" | "HYBRID" | "SYNTHETIC"
    drainage_quality: str  # "EXCELLENT" | "GOOD" | "POOR"
    typical_grass_mm: float  # grass cutting length (mm)
    possession_home_index: float = 0.50  # 0=direct, 1=tiki-taka

    @property
    def area_m2(self) -> float:
        return self.length_m * self.width_m

    @property
    def is_narrow(self) -> bool:
        return self.width_m < 66.0

    @property
    def is_wide(self) -> bool:
        return self.width_m > 70.0


# 55 stadiums across top leagues
STADIUM_TURF_DB: Dict[str, TurfProfile] = {
    # Premier League
    "anfield": TurfProfile("anfield", 101, 68, "HYBRID", "GOOD", 28, 0.58),
    "etihad": TurfProfile("etihad", 105, 68, "HYBRID", "EXCELLENT", 24, 0.82),
    "old_trafford": TurfProfile("old_trafford", 105, 68, "HYBRID", "GOOD", 30, 0.55),
    "stamford_bridge": TurfProfile(
        "stamford_bridge", 103, 67, "HYBRID", "GOOD", 30, 0.60
    ),
    "emirates": TurfProfile("emirates", 105, 68, "HYBRID", "GOOD", 28, 0.65),
    "tottenham_hotspur": TurfProfile(
        "tottenham_hotspur", 105, 68, "HYBRID", "EXCELLENT", 26, 0.62
    ),
    "villa_park": TurfProfile("villa_park", 100, 68, "NATURAL", "GOOD", 32, 0.50),
    "st_james_park": TurfProfile("st_james_park", 105, 68, "NATURAL", "GOOD", 35, 0.45),
    "goodison_park": TurfProfile("goodison_park", 100, 68, "NATURAL", "POOR", 38, 0.42),
    "elland_road": TurfProfile("elland_road", 105, 68, "NATURAL", "GOOD", 33, 0.48),
    "molineux": TurfProfile("molineux", 100, 65, "HYBRID", "GOOD", 30, 0.48),
    "king_power": TurfProfile("king_power", 100, 67, "NATURAL", "GOOD", 32, 0.50),
    # La Liga
    "santiago_bernabeu": TurfProfile(
        "santiago_bernabeu", 105, 68, "HYBRID", "EXCELLENT", 26, 0.60
    ),
    "camp_nou": TurfProfile("camp_nou", 105, 68, "HYBRID", "EXCELLENT", 24, 0.80),
    "metropolitano": TurfProfile("metropolitano", 105, 68, "HYBRID", "GOOD", 30, 0.42),
    "mestalla": TurfProfile("mestalla", 105, 66, "NATURAL", "GOOD", 32, 0.55),
    "san_mames": TurfProfile("san_mames", 105, 68, "NATURAL", "GOOD", 34, 0.48),
    "estadio_benito_villamarin": TurfProfile(
        "estadio_benito_villamarin", 105, 68, "HYBRID", "GOOD", 28, 0.58
    ),
    # Serie A
    "san_siro": TurfProfile("san_siro", 105, 68, "NATURAL", "POOR", 40, 0.55),
    "olimpico_roma": TurfProfile("olimpico_roma", 105, 68, "NATURAL", "POOR", 42, 0.52),
    "juventus_stadium": TurfProfile(
        "juventus_stadium", 105, 68, "HYBRID", "EXCELLENT", 26, 0.62
    ),
    "san_paolo": TurfProfile("san_paolo", 105, 68, "NATURAL", "GOOD", 35, 0.58),
    "gewiss_stadium": TurfProfile(
        "gewiss_stadium", 105, 68, "HYBRID", "GOOD", 28, 0.60
    ),
    # Bundesliga
    "allianz_arena": TurfProfile(
        "allianz_arena", 105, 68, "HYBRID", "EXCELLENT", 24, 0.78
    ),
    "signal_iduna_park": TurfProfile(
        "signal_iduna_park", 105, 68, "NATURAL", "EXCELLENT", 28, 0.65
    ),
    "red_bull_arena": TurfProfile(
        "red_bull_arena", 105, 68, "HYBRID", "EXCELLENT", 26, 0.68
    ),
    "bayarena": TurfProfile("bayarena", 105, 68, "NATURAL", "GOOD", 30, 0.60),
    "volksparkstadion": TurfProfile(
        "volksparkstadion", 105, 68, "NATURAL", "GOOD", 32, 0.55
    ),
    # Ligue 1
    "parc_des_princes": TurfProfile(
        "parc_des_princes", 105, 68, "HYBRID", "EXCELLENT", 26, 0.65
    ),
    "velodrome": TurfProfile("velodrome", 105, 68, "HYBRID", "GOOD", 30, 0.52),
    # Scandinavia / lower leagues — synthetic turf common
    "ullevaal": TurfProfile("ullevaal", 105, 68, "SYNTHETIC", "EXCELLENT", 0, 0.50),
    "telia_parken": TurfProfile(
        "telia_parken", 105, 65, "SYNTHETIC", "EXCELLENT", 0, 0.52
    ),
    "friends_arena": TurfProfile(
        "friends_arena", 105, 68, "SYNTHETIC", "EXCELLENT", 0, 0.55
    ),
    # Scotland
    "celtic_park": TurfProfile("celtic_park", 105, 68, "NATURAL", "GOOD", 35, 0.58),
    "ibrox": TurfProfile("ibrox", 105, 68, "NATURAL", "GOOD", 33, 0.55),
    # Portugal
    "estadio_da_luz": TurfProfile(
        "estadio_da_luz", 105, 68, "HYBRID", "EXCELLENT", 26, 0.65
    ),
    "estadio_dragao": TurfProfile(
        "estadio_dragao", 105, 68, "HYBRID", "GOOD", 28, 0.62
    ),
    # Netherlands
    "johan_cruyff_arena": TurfProfile(
        "johan_cruyff_arena", 105, 68, "HYBRID", "EXCELLENT", 26, 0.68
    ),
    # Turkey
    "sukru_saracoglu": TurfProfile(
        "sukru_saracoglu", 105, 68, "NATURAL", "GOOD", 35, 0.48
    ),
    # High altitude / South America
    "estadio_azteca": TurfProfile(
        "estadio_azteca", 105, 68, "NATURAL", "GOOD", 40, 0.50
    ),
    "estadio_hernando_siles": TurfProfile(
        "estadio_hernando_siles", 100, 64, "NATURAL", "POOR", 48, 0.40
    ),
    "estadio_atahualpa": TurfProfile(
        "estadio_atahualpa", 100, 65, "NATURAL", "POOR", 45, 0.42
    ),
    "estadio_mineirao": TurfProfile(
        "estadio_mineirao", 105, 68, "NATURAL", "GOOD", 32, 0.55
    ),
    "estadio_monumental": TurfProfile(
        "estadio_monumental", 105, 68, "NATURAL", "GOOD", 30, 0.60
    ),
}

_STANDARD_AREA_M2 = 105.0 * 68.0  # 7140 m²
_STANDARD_GRASS_MM = 28.0


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class TurfCondition:
    """Combined static profile + live weather state."""

    profile: TurfProfile
    rain_mm_per_hour: float = 0.0
    wind_speed_kph: float = 0.0
    temperature_c: float = 15.0
    ball_speed_multiplier: float = field(init=False)
    effective_grass_mm: float = field(init=False)
    surface_label: str = field(init=False)

    def __post_init__(self) -> None:
        # Rain accelerates ball on short/wet grass
        rain_factor = min(self.rain_mm_per_hour / 8.0, 1.0)
        self.ball_speed_multiplier = 1.0 + rain_factor * 0.20

        # Rain slightly compresses grass
        self.effective_grass_mm = max(
            self.profile.typical_grass_mm * (1.0 - rain_factor * 0.10), 10.0
        )

        # Surface label
        if self.ball_speed_multiplier > 1.12:
            self.surface_label = "FAST_WET"
        elif self.profile.typical_grass_mm > 38:
            self.surface_label = "SLOW"
        elif self.profile.typical_grass_mm < 27:
            self.surface_label = "FAST_DRY"
        else:
            self.surface_label = "NORMAL"


@dataclass
class TeamTurfProfile:
    """A team's style and its turf-compatibility."""

    team: str
    possession_index: float = 0.50  # 0=direct, 1=tiki-taka
    pressing_intensity: float = 0.50  # high press
    direct_play_index: float = 0.50  # long-ball / aerial


@dataclass
class TurfImpact:
    """Full turf impact assessment for one match."""

    home_team: str
    away_team: str
    condition: TurfCondition
    lambda_home_multiplier: float
    lambda_away_multiplier: float
    xg_area_multiplier: float
    ball_speed_multiplier: float
    grass_speed_factor: float
    home_possession_bonus: float
    turf_type_effect: str
    notes: List[str]

    def summary(self) -> str:
        lines = [
            "=" * 52,
            f"TURF IMPACT: {self.home_team} vs {self.away_team}",
            f"Stadium: {self.condition.profile.stadium_slug}",
            f"Surface: {self.condition.condition.surface_label if hasattr(self.condition, 'condition') else self.condition.surface_label}",
            f"Turf type: {self.condition.profile.turf_type} | Grass: {self.condition.effective_grass_mm:.0f}mm",
            f"Pitch: {self.condition.profile.length_m}×{self.condition.profile.width_m}m",
            "-" * 52,
            f"Ball speed mult : ×{self.ball_speed_multiplier:.3f}",
            f"Grass speed fac : ×{self.grass_speed_factor:.3f}",
            f"xG area mult    : ×{self.xg_area_multiplier:.3f}",
            f"Home poss bonus : +{self.home_possession_bonus:.3f}",
            f"Turf type effect: {self.turf_type_effect}",
            "-" * 52,
            f"λ home multiplier: ×{self.lambda_home_multiplier:.3f}",
            f"λ away multiplier: ×{self.lambda_away_multiplier:.3f}",
        ]
        if self.notes:
            lines.append("-" * 52)
            for note in self.notes:
                lines.append(f"  • {note}")
        lines.append("=" * 52)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class TurfEngine:
    """
    Analyses pitch & turf conditions and computes λ multipliers.

    Parameters
    ----------
    team_profiles : dict mapping team slug to TeamTurfProfile.
        If a team is not found, neutral defaults are used.
    """

    _KNOWN_TEAMS: Dict[str, TeamTurfProfile] = {
        "manchester city": TeamTurfProfile("Manchester City", 0.85, 0.80, 0.20),
        "barcelona": TeamTurfProfile("Barcelona", 0.88, 0.75, 0.15),
        "real madrid": TeamTurfProfile("Real Madrid", 0.70, 0.65, 0.35),
        "atletico madrid": TeamTurfProfile("Atletico Madrid", 0.40, 0.70, 0.55),
        "arsenal": TeamTurfProfile("Arsenal", 0.72, 0.72, 0.30),
        "liverpool": TeamTurfProfile("Liverpool", 0.65, 0.85, 0.40),
        "chelsea": TeamTurfProfile("Chelsea", 0.62, 0.70, 0.40),
        "juventus": TeamTurfProfile("Juventus", 0.50, 0.55, 0.55),
        "inter milan": TeamTurfProfile("Inter Milan", 0.55, 0.65, 0.50),
        "ac milan": TeamTurfProfile("AC Milan", 0.58, 0.62, 0.48),
        "borussia dortmund": TeamTurfProfile("Borussia Dortmund", 0.65, 0.80, 0.42),
        "napoli": TeamTurfProfile("Napoli", 0.70, 0.72, 0.35),
        "psg": TeamTurfProfile("PSG", 0.68, 0.70, 0.38),
        "stoke city": TeamTurfProfile("Stoke City", 0.25, 0.55, 0.85),
        "burnley": TeamTurfProfile("Burnley", 0.30, 0.60, 0.80),
    }

    def __init__(
        self, team_profiles: Optional[Dict[str, TeamTurfProfile]] = None
    ) -> None:
        self._custom_profiles: Dict[str, TeamTurfProfile] = {}
        if team_profiles:
            for k, v in team_profiles.items():
                self._custom_profiles[k.lower()] = v

    def analyse(
        self,
        venue_slug: str,
        home_team: str,
        away_team: str,
        rain_mm_per_hour: float = 0.0,
        wind_kph: float = 0.0,
        temperature_c: float = 15.0,
    ) -> TurfImpact:
        """Compute full turf impact for a fixture."""
        profile = self._get_profile(venue_slug)
        if profile is None:
            profile = TurfProfile(
                stadium_slug=venue_slug,
                length_m=105.0,
                width_m=68.0,
                turf_type="NATURAL",
                drainage_quality="GOOD",
                typical_grass_mm=30.0,
                possession_home_index=0.50,
            )
            logger.warning("Venue '%s' not in DB — using defaults.", venue_slug)

        condition = TurfCondition(
            profile=profile,
            rain_mm_per_hour=rain_mm_per_hour,
            wind_speed_kph=wind_kph,
            temperature_c=temperature_c,
        )

        home_tp = self._resolve_team(home_team)
        away_tp = self._resolve_team(away_team)

        notes: List[str] = []

        # xG area multiplier
        area_ratio = profile.area_m2 / _STANDARD_AREA_M2
        xg_area_mult = 0.85 + 0.30 * area_ratio
        if profile.is_narrow:
            notes.append(f"Narrow pitch ({profile.width_m}m) — Under edge")
        if profile.is_wide:
            notes.append(f"Wide pitch ({profile.width_m}m) — Over edge")

        # Grass speed factor
        grass_speed = _STANDARD_GRASS_MM / max(condition.effective_grass_mm, 10.0)
        if condition.effective_grass_mm > 38:
            notes.append(
                f"Long grass ({condition.effective_grass_mm:.0f}mm) — slows technical play"
            )

        # Synthetic turf effect
        turf_label = "Neutral (natural/hybrid)"
        synthetic_home = 0.0
        synthetic_away = 0.0
        if profile.turf_type == "SYNTHETIC":
            synthetic_home = (
                home_tp.direct_play_index * 0.05 - home_tp.possession_index * 0.08
            )
            synthetic_away = (
                away_tp.direct_play_index * 0.05 - away_tp.possession_index * 0.08
            )
            turf_label = "Synthetic: physical teams +5%, technical teams -8%"
            notes.append("Synthetic turf — dribblers penalised, aerial play rewarded")

        # Home possession bonus (wet ball speed + home possession style)
        if (
            condition.ball_speed_multiplier > 1.10
            and profile.possession_home_index > 0.65
        ):
            home_poss_bonus = (
                (condition.ball_speed_multiplier - 1.0)
                * profile.possession_home_index
                * 0.5
            )
            notes.append(
                f"Wet ball +home possession combo: home gets +{home_poss_bonus:.2f}λ bonus"
            )
        else:
            home_poss_bonus = 0.0

        if condition.ball_speed_multiplier > 1.10:
            notes.append(
                f"Wet/fast ball (×{condition.ball_speed_multiplier:.2f}) — benefits possession sides"
            )

        # Compose multipliers
        home_mult = max(
            0.65,
            xg_area_mult * grass_speed * (1.0 + synthetic_home + home_poss_bonus),
        )
        away_mult = max(
            0.65,
            xg_area_mult * grass_speed * (1.0 + synthetic_away),
        )

        # Drainage effect: POOR in rain increases surface water → slows game
        if condition.rain_mm_per_hour > 2.0 and profile.drainage_quality == "POOR":
            drainage_penalty = 0.92
            home_mult *= drainage_penalty
            away_mult *= drainage_penalty
            notes.append("Poor drainage in rain — heavy surface, slower ball overall")

        logger.info(
            "TurfEngine: %s@%s rain=%.1fmm/h → lh×%.3f la×%.3f",
            home_team,
            venue_slug,
            rain_mm_per_hour,
            home_mult,
            away_mult,
        )

        return TurfImpact(
            home_team=home_team,
            away_team=away_team,
            condition=condition,
            lambda_home_multiplier=home_mult,
            lambda_away_multiplier=away_mult,
            xg_area_multiplier=xg_area_mult,
            ball_speed_multiplier=condition.ball_speed_multiplier,
            grass_speed_factor=grass_speed,
            home_possession_bonus=home_poss_bonus,
            turf_type_effect=turf_label,
            notes=notes,
        )

    def adjust_lambdas(
        self,
        lambda_home: float,
        lambda_away: float,
        impact: TurfImpact,
    ) -> Tuple[float, float]:
        """Apply turf multipliers to pre-match expected-goal rates."""
        adj_h = max(0.10, lambda_home * impact.lambda_home_multiplier)
        adj_a = max(0.10, lambda_away * impact.lambda_away_multiplier)
        return adj_h, adj_a

    def find_edge_venues(self, condition_threshold: float = 0.10) -> List[str]:
        """
        Return venue slugs where the turf creates a systematic market edge
        (multiplier deviates from 1.0 by more than *condition_threshold*).
        """
        edges: List[str] = []
        for slug, profile in STADIUM_TURF_DB.items():
            area_ratio = profile.area_m2 / _STANDARD_AREA_M2
            xg = 0.85 + 0.30 * area_ratio
            grass = _STANDARD_GRASS_MM / max(profile.typical_grass_mm, 10.0)
            mult = xg * grass
            if (
                abs(mult - 1.0) > condition_threshold
                or profile.turf_type == "SYNTHETIC"
            ):
                edges.append(slug)
        return sorted(edges)

    def _get_profile(self, slug: str) -> Optional[TurfProfile]:
        key = slug.lower().replace(" ", "_").replace("-", "_")
        return STADIUM_TURF_DB.get(key)

    def _resolve_team(self, name: str) -> TeamTurfProfile:
        key = name.lower()
        if key in self._custom_profiles:
            return self._custom_profiles[key]
        return self._KNOWN_TEAMS.get(key, TeamTurfProfile(name))


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------


def quick_turf_check(
    venue_slug: str,
    home_team: str = "home",
    away_team: str = "away",
    rain_mm_per_hour: float = 0.0,
) -> TurfImpact:
    """Fast-path turf assessment with no configuration."""
    return TurfEngine().analyse(
        venue_slug=venue_slug,
        home_team=home_team,
        away_team=away_team,
        rain_mm_per_hour=rain_mm_per_hour,
    )

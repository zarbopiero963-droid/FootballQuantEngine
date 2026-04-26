"""
lineup_sniper.py
================
Intercepts official lineup announcements (~60 min before kickoff), compares them
to the expected lineup, recalculates the affected team's expected-goals contribution,
and emits a betting alert — before the bookmaker's human trader can react.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_KEY_PLAYER_THRESHOLD = 0.15  # fraction of team xG; above this = "key player"
_MAJOR_IMPACT_THRESHOLD = 0.25  # above this = major impact (strong alert)
_DEFAULT_SQUAD_SIZE = 11


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class PlayerProfile:
    """A player's contribution profile within a team."""

    name: str
    team: str
    position: str  # "GK" | "DEF" | "MID" | "FWD"
    xg_share: float  # fraction of team total xG this player contributes (0–1)
    xa_share: float  # fraction of team xA this player contributes (0–1)
    elo_impact: float  # Elo points the team loses when player is absent
    is_key: bool = field(init=False)

    def __post_init__(self) -> None:
        self.is_key = self.xg_share >= _KEY_PLAYER_THRESHOLD or self.elo_impact >= 30


@dataclass
class ExpectedLineup:
    """Pre-match expected starting eleven for a team."""

    team: str
    fixture_id: int
    players: List[PlayerProfile]  # expected starters (11 players)
    total_xg: float  # team total xG estimate before lineup known
    total_elo: float  # team Elo before lineup known


@dataclass
class OfficialLineup:
    """Official confirmed lineup (announced ~60 min before kickoff)."""

    team: str
    fixture_id: int
    confirmed_players: List[str]  # names of confirmed starters
    announced_at: float  # unix timestamp of announcement


@dataclass
class LineupDiff:
    """Difference between expected and official lineup."""

    fixture_id: int
    team: str
    missing_key_players: List[PlayerProfile]  # expected but not in official
    surprise_starters: List[str]  # in official but not expected
    xg_delta: float  # how much team xG changes (negative = worse)
    elo_delta: float  # Elo points change
    impact_level: str  # "NONE" | "MINOR" | "MAJOR" | "CRITICAL"
    announced_at: float

    def __str__(self) -> str:
        return (
            f"LineupDiff [{self.team}] fx={self.fixture_id} "
            f"impact={self.impact_level} xG_delta={self.xg_delta:+.3f} "
            f"elo_delta={self.elo_delta:+.1f} "
            f"missing={[p.name for p in self.missing_key_players]}"
        )


@dataclass
class LineupAlert:
    """Betting alert triggered by a significant lineup difference."""

    fixture_id: int
    affected_team: str  # the team with missing players
    opponent_team: str  # team to consider betting on
    diff: LineupDiff
    adj_lambda: float  # adjusted lambda for the affected team
    original_lambda: float  # pre-adjustment lambda
    recommended_action: str  # e.g. "BET AGAINST Arsenal" / "FADE Liverpool home"
    confidence: str  # "HIGH" | "MEDIUM" | "LOW"
    timestamp: float

    def __str__(self) -> str:
        return (
            f"LINEUP ALERT [fx={self.fixture_id}] {self.recommended_action} "
            f"λ: {self.original_lambda:.3f}→{self.adj_lambda:.3f} "
            f"conf={self.confidence} impact={self.diff.impact_level}"
        )


# ---------------------------------------------------------------------------
# LineupSniper
# ---------------------------------------------------------------------------


class LineupSniper:
    """
    Monitors expected vs official lineups and triggers betting alerts.

    Workflow:
      1. Register expected lineups via register_expected().
      2. When official lineup arrives, call process_official().
      3. If key players are absent, get_alerts() returns LineupAlert objects.

    Lambda adjustment model:
      adj_lambda = original_lambda × (1 - xg_delta_fraction)
      where xg_delta_fraction = abs(xg_delta) / total_team_xg
      Minimum adj_lambda = 0.20 (can't collapse to zero).

    Elo adjustment:
      adj_elo = original_elo + elo_delta
    """

    def __init__(
        self,
        on_alert: Optional[Callable[[LineupAlert], None]] = None,
    ) -> None:
        self._expected: Dict[Tuple[int, str], ExpectedLineup] = {}
        self._alerts: List[LineupAlert] = []
        self._on_alert = on_alert

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register_expected(self, lineup: ExpectedLineup) -> None:
        """Store the expected lineup for a fixture+team."""
        key = (lineup.fixture_id, lineup.team.lower())
        self._expected[key] = lineup
        logger.debug(
            "Registered expected lineup for %s fx=%d (%d players, total_xg=%.3f)",
            lineup.team,
            lineup.fixture_id,
            len(lineup.players),
            lineup.total_xg,
        )

    def process_official(
        self,
        official: OfficialLineup,
        opponent_team: str,
        original_lambda: float,
    ) -> Optional[LineupAlert]:
        """
        Compare official lineup to expected. Return LineupAlert if key players absent,
        else None.

        Steps:
          1. Look up expected lineup for this (fixture_id, team).
          2. Find missing key players (in expected but not in official, by name match).
          3. Compute xg_delta = -sum(missing.xg_share × total_xg) for key players.
          4. Compute elo_delta = -sum(missing.elo_impact).
          5. Classify impact: CRITICAL if |xg_delta/total_xg| > 0.40,
             MAJOR if > 0.25, MINOR if > 0.10, NONE otherwise.
          6. If impact != NONE, build LineupAlert and call on_alert callback.
        """
        key = (official.fixture_id, official.team.lower())
        expected = self._expected.get(key)
        if expected is None:
            logger.warning(
                "No expected lineup registered for %s fx=%d — skipping diff.",
                official.team,
                official.fixture_id,
            )
            return None

        # Normalise confirmed names once for O(1) lookups
        confirmed_normalised: Set[str] = {
            n.strip().lower() for n in official.confirmed_players
        }

        # Step 2: find key players expected but absent from official lineup
        missing_key_players: List[PlayerProfile] = []
        expected_names: Set[str] = set()
        for player in expected.players:
            normalised_name = player.name.strip().lower()
            expected_names.add(normalised_name)
            if normalised_name not in confirmed_normalised and player.is_key:
                missing_key_players.append(player)
                logger.info(
                    "Key player absent: %s (%s) — xg_share=%.3f elo_impact=%.1f",
                    player.name,
                    player.position,
                    player.xg_share,
                    player.elo_impact,
                )

        # Surprise starters: in official but not in expected
        surprise_starters: List[str] = [
            name
            for name in official.confirmed_players
            if name.strip().lower() not in expected_names
        ]
        if surprise_starters:
            logger.debug(
                "Surprise starters for %s: %s", official.team, surprise_starters
            )

        # Step 3: xg_delta (negative — team loses attack output)
        total_xg = expected.total_xg
        xg_delta: float = -sum(p.xg_share * total_xg for p in missing_key_players)

        # Step 4: elo_delta (negative — team weakens)
        elo_delta: float = -sum(p.elo_impact for p in missing_key_players)

        # Step 5: classify impact
        impact = self._impact_level(xg_delta, total_xg)

        diff = LineupDiff(
            fixture_id=official.fixture_id,
            team=official.team,
            missing_key_players=missing_key_players,
            surprise_starters=surprise_starters,
            xg_delta=xg_delta,
            elo_delta=elo_delta,
            impact_level=impact,
            announced_at=official.announced_at,
        )
        logger.debug("LineupDiff computed: %s", diff)

        # Step 6: only alert when there is meaningful impact
        if impact == "NONE":
            logger.debug(
                "Impact level NONE for %s fx=%d — no alert emitted.",
                official.team,
                official.fixture_id,
            )
            return None

        adj_lambda = self.adjust_lambda(original_lambda, xg_delta, total_xg)
        confidence = self._confidence(diff)
        action = self._recommended_action(diff, opponent_team)

        alert = LineupAlert(
            fixture_id=official.fixture_id,
            affected_team=official.team,
            opponent_team=opponent_team,
            diff=diff,
            adj_lambda=adj_lambda,
            original_lambda=original_lambda,
            recommended_action=action,
            confidence=confidence,
            timestamp=time.time(),
        )
        self._alerts.append(alert)
        logger.warning("LineupAlert emitted: %s", alert)

        if self._on_alert is not None:
            try:
                self._on_alert(alert)
            except Exception:  # noqa: BLE001
                logger.exception(
                    "on_alert callback raised an exception for fx=%d",
                    official.fixture_id,
                )

        return alert

    def get_alerts(self, fixture_id: Optional[int] = None) -> List[LineupAlert]:
        """Return all alerts, optionally filtered to one fixture."""
        if fixture_id is None:
            return list(self._alerts)
        return [a for a in self._alerts if a.fixture_id == fixture_id]

    # ------------------------------------------------------------------
    # Lambda / Elo helpers
    # ------------------------------------------------------------------

    def adjust_lambda(
        self,
        original_lambda: float,
        xg_delta: float,
        total_team_xg: float,
    ) -> float:
        """
        adj_lambda = original_lambda × max(0.40, 1 - abs(xg_delta) / max(total_team_xg, 0.01))
        Minimum: 0.20.
        """
        safe_total = max(total_team_xg, 0.01)
        fraction = abs(xg_delta) / safe_total
        multiplier = max(0.40, 1.0 - fraction)
        adj = original_lambda * multiplier
        adj = max(adj, 0.20)
        logger.debug(
            "adjust_lambda: %.3f × %.4f = %.3f (floored at 0.20 → %.3f)",
            original_lambda,
            multiplier,
            original_lambda * multiplier,
            adj,
        )
        return adj

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _confidence(self, diff: LineupDiff) -> str:
        """HIGH if CRITICAL, MEDIUM if MAJOR, LOW if MINOR."""
        mapping = {
            "CRITICAL": "HIGH",
            "MAJOR": "MEDIUM",
            "MINOR": "LOW",
            "NONE": "LOW",
        }
        return mapping.get(diff.impact_level, "LOW")

    def _recommended_action(self, diff: LineupDiff, opponent: str) -> str:
        """
        Build a human-readable betting directive.

        CRITICAL  → "BET AGAINST {team} — missing {names}"
        MAJOR     → "BET AGAINST {team} — missing {names}"
        MINOR     → "FADE {team} — missing {names}"
        """
        missing_names = ", ".join(p.name for p in diff.missing_key_players) or "unknown"
        team = diff.team

        if diff.impact_level in ("CRITICAL", "MAJOR"):
            return f"BET AGAINST {team} — missing {missing_names}"
        # MINOR
        return f"FADE {team} — missing {missing_names}"

    def _impact_level(self, xg_delta: float, total_xg: float) -> str:
        """
        Classify impact as NONE / MINOR / MAJOR / CRITICAL.

        Thresholds on |xg_delta / total_xg|:
          > 0.40 → CRITICAL
          > 0.25 → MAJOR
          > 0.10 → MINOR
          else   → NONE
        """
        safe_total = max(total_xg, 0.01)
        ratio = abs(xg_delta) / safe_total
        if ratio > 0.40:
            return "CRITICAL"
        if ratio > 0.25:
            return "MAJOR"
        if ratio > 0.10:
            return "MINOR"
        return "NONE"


# ---------------------------------------------------------------------------
# Module-level roster builder helpers
# ---------------------------------------------------------------------------


def build_player_profile(
    name: str,
    team: str,
    position: str,
    xg_share: float,
    xa_share: float = 0.0,
    elo_impact: float = 0.0,
) -> PlayerProfile:
    """Convenience constructor for PlayerProfile."""
    profile = PlayerProfile(
        name=name,
        team=team,
        position=position,
        xg_share=xg_share,
        xa_share=xa_share,
        elo_impact=elo_impact,
    )
    logger.debug(
        "Built PlayerProfile: %s (%s/%s) xg_share=%.3f is_key=%s",
        name,
        team,
        position,
        xg_share,
        profile.is_key,
    )
    return profile


def expected_lineup_from_dict(data: dict) -> ExpectedLineup:
    """
    Build ExpectedLineup from a dict with keys:
      team, fixture_id, total_xg, total_elo,
      players: list of {name, position, xg_share, xa_share, elo_impact}

    All player fields except name and position default to 0.0 if absent.
    """
    team: str = data["team"]
    fixture_id: int = int(data["fixture_id"])
    total_xg: float = float(data["total_xg"])
    total_elo: float = float(data["total_elo"])

    players: List[PlayerProfile] = []
    for raw in data.get("players", []):
        profile = build_player_profile(
            name=raw["name"],
            team=team,
            position=raw["position"],
            xg_share=float(raw.get("xg_share", 0.0)),
            xa_share=float(raw.get("xa_share", 0.0)),
            elo_impact=float(raw.get("elo_impact", 0.0)),
        )
        players.append(profile)

    lineup = ExpectedLineup(
        team=team,
        fixture_id=fixture_id,
        players=players,
        total_xg=total_xg,
        total_elo=total_elo,
    )
    logger.debug(
        "expected_lineup_from_dict: %s fx=%d %d players total_xg=%.3f",
        team,
        fixture_id,
        len(players),
        total_xg,
    )
    return lineup


def official_lineup_from_dict(data: dict) -> OfficialLineup:
    """
    Build OfficialLineup from a dict with keys:
      team, fixture_id, confirmed_players (list of str)
    announced_at defaults to time.time() if not provided.
    """
    team: str = data["team"]
    fixture_id: int = int(data["fixture_id"])
    confirmed_players: List[str] = list(data.get("confirmed_players", []))
    announced_at: float = float(data.get("announced_at", time.time()))

    lineup = OfficialLineup(
        team=team,
        fixture_id=fixture_id,
        confirmed_players=confirmed_players,
        announced_at=announced_at,
    )
    logger.debug(
        "official_lineup_from_dict: %s fx=%d %d confirmed players",
        team,
        fixture_id,
        len(confirmed_players),
    )
    return lineup

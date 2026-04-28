"""expected_threat.py — Expected Threat (xT) model for football pitch analysis.

Based on Karun Singh's 2018 xT framework.  The pitch is divided into a 16×12
grid (16 columns defence-to-attack, 12 rows bottom-to-top).  Every ball touch
or pass is valued by the change in xT it produces.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default xT grid — 12 rows × 16 columns
# Derived from published xT research (Singh 2018, open StatsBomb data).
# Row index 0 = bottom (south touchline), 11 = top (north touchline).
# Column index 0 = own goal end, 15 = opponent goal mouth.
# Values increase smoothly left-to-right; central rows (4–7) peak higher.
# ---------------------------------------------------------------------------
_DEFAULT_XT_GRID: List[List[float]] = [
    # row 0 — bottom (south) touchline
    [
        0.001,
        0.002,
        0.002,
        0.003,
        0.003,
        0.004,
        0.005,
        0.006,
        0.008,
        0.010,
        0.014,
        0.020,
        0.028,
        0.040,
        0.060,
        0.090,
    ],
    # row 1
    [
        0.001,
        0.002,
        0.003,
        0.004,
        0.004,
        0.005,
        0.007,
        0.009,
        0.012,
        0.016,
        0.022,
        0.032,
        0.048,
        0.072,
        0.110,
        0.150,
    ],
    # row 2
    [
        0.002,
        0.003,
        0.004,
        0.005,
        0.006,
        0.007,
        0.009,
        0.012,
        0.016,
        0.022,
        0.032,
        0.050,
        0.080,
        0.130,
        0.200,
        0.280,
    ],
    # row 3
    [
        0.002,
        0.003,
        0.004,
        0.006,
        0.007,
        0.009,
        0.012,
        0.016,
        0.022,
        0.032,
        0.050,
        0.082,
        0.140,
        0.210,
        0.320,
        0.420,
    ],
    # row 4 — entering the productive central band
    [
        0.003,
        0.004,
        0.005,
        0.007,
        0.009,
        0.012,
        0.017,
        0.023,
        0.032,
        0.048,
        0.075,
        0.120,
        0.190,
        0.290,
        0.420,
        0.560,
    ],
    # row 5
    [
        0.003,
        0.005,
        0.006,
        0.008,
        0.011,
        0.015,
        0.021,
        0.030,
        0.042,
        0.062,
        0.095,
        0.150,
        0.230,
        0.340,
        0.490,
        0.640,
    ],
    # row 6 — central, highest threat band
    [
        0.003,
        0.005,
        0.006,
        0.008,
        0.011,
        0.015,
        0.021,
        0.030,
        0.042,
        0.062,
        0.095,
        0.150,
        0.230,
        0.340,
        0.490,
        0.640,
    ],
    # row 7
    [
        0.003,
        0.004,
        0.005,
        0.007,
        0.009,
        0.012,
        0.017,
        0.023,
        0.032,
        0.048,
        0.075,
        0.120,
        0.190,
        0.290,
        0.420,
        0.560,
    ],
    # row 8
    [
        0.002,
        0.003,
        0.004,
        0.006,
        0.007,
        0.009,
        0.012,
        0.016,
        0.022,
        0.032,
        0.050,
        0.082,
        0.140,
        0.210,
        0.320,
        0.420,
    ],
    # row 9
    [
        0.002,
        0.003,
        0.004,
        0.005,
        0.006,
        0.007,
        0.009,
        0.012,
        0.016,
        0.022,
        0.032,
        0.050,
        0.080,
        0.130,
        0.200,
        0.280,
    ],
    # row 10
    [
        0.001,
        0.002,
        0.003,
        0.004,
        0.004,
        0.005,
        0.007,
        0.009,
        0.012,
        0.016,
        0.022,
        0.032,
        0.048,
        0.072,
        0.110,
        0.150,
    ],
    # row 11 — top (north) touchline
    [
        0.001,
        0.002,
        0.002,
        0.003,
        0.003,
        0.004,
        0.005,
        0.006,
        0.008,
        0.010,
        0.014,
        0.020,
        0.028,
        0.040,
        0.060,
        0.090,
    ],
]

# Validate shape at import time so misconfiguration is caught immediately.
assert len(_DEFAULT_XT_GRID) == 12, "xT grid must have exactly 12 rows"
assert all(
    len(r) == 16 for r in _DEFAULT_XT_GRID
), "each row must have exactly 16 columns"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class PitchZone:
    """One cell of the 16×12 xT grid."""

    col: int  # 0–15 (defence → attack)
    row: int  # 0–11 (bottom → top)
    xt_value: float
    x_centre: float  # metres along pitch length
    y_centre: float  # metres along pitch width


@dataclass
class TouchEvent:
    """A single ball touch or pass."""

    player: str
    team: str
    from_x: float  # metres
    from_y: float
    to_x: float  # metres (same as from_x if a stationary touch)
    to_y: float
    action: str  # "pass" | "carry" | "shot" | "dribble"
    minute: int
    outcome: str  # "success" | "fail"
    timestamp: float = 0.0

    @property
    def is_successful(self) -> bool:
        return self.outcome == "success"

    @property
    def xt_gain(self) -> float:
        """Populated by XTModel.evaluate_touch(); 0.0 before evaluation."""
        return getattr(self, "_xt_gain", 0.0)


@dataclass
class XTReport:
    """xT analysis for one team in one game."""

    team: str
    total_xt: float  # sum of all positive xT gains
    total_xt_conceded: float  # opponent's total positive xT
    xt_per_touch: float  # total_xt / n_touches
    xt_dominance: float  # total_xt / (total_xt + total_xt_conceded)
    top_zones: List[Tuple[int, int, float]]  # (col, row, total_xt) — top 5 zones
    top_players: List[Tuple[str, float]]  # (player, total_xt_contribution) — top 5
    n_touches: int
    n_successful_passes: int

    def __str__(self) -> str:
        return (
            f"xT Report [{self.team}] "
            f"xT={self.total_xt:.3f} "
            f"per_touch={self.xt_per_touch:.4f} "
            f"dominance={self.xt_dominance:.1%}"
        )


# ---------------------------------------------------------------------------
# XTModel
# ---------------------------------------------------------------------------


class XTModel:
    """
    Expected Threat model for football pitch analysis.

    The model assigns an xT value to each zone of the pitch.  When a player
    moves the ball from zone A to zone B (pass or carry), the xT gain is::

        xt_gain = xt(B) - xt(A)       if action is successful
        xt_gain = -xt(A) × 0.5        if action fails (possession lost)

    Parameters
    ----------
    xt_grid:
        12×16 nested list of xT values.  Defaults to ``_DEFAULT_XT_GRID``.
    """

    PITCH_LENGTH: float = 105.0
    PITCH_WIDTH: float = 68.0
    COLS: int = 16
    ROWS: int = 12

    def __init__(self, xt_grid: Optional[List[List[float]]] = None) -> None:
        self._grid: List[List[float]] = (
            xt_grid if xt_grid is not None else _DEFAULT_XT_GRID
        )

        if len(self._grid) != self.ROWS or any(len(r) != self.COLS for r in self._grid):
            raise ValueError(
                f"xt_grid must be {self.ROWS} rows × {self.COLS} columns, "
                f"got {len(self._grid)} rows"
            )

        # Build zone lookup once
        self._zones: Dict[Tuple[int, int], PitchZone] = {}
        for row in range(self.ROWS):
            for col in range(self.COLS):
                x_c = (col + 0.5) * self.PITCH_LENGTH / self.COLS
                y_c = (row + 0.5) * self.PITCH_WIDTH / self.ROWS
                self._zones[(col, row)] = PitchZone(
                    col=col,
                    row=row,
                    xt_value=self._grid[row][col],
                    x_centre=round(x_c, 4),
                    y_centre=round(y_c, 4),
                )

        logger.debug(
            "XTModel initialised: %d×%d grid, max_xt=%.4f",
            self.COLS,
            self.ROWS,
            max(v for row in self._grid for v in row),
        )

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------

    def _xy_to_col_row(self, x: float, y: float) -> Tuple[int, int]:
        """Convert pitch metres to (col, row) with boundary clamping."""
        col = int(x / self.PITCH_LENGTH * self.COLS)
        row = int(y / self.PITCH_WIDTH * self.ROWS)
        col = max(0, min(col, self.COLS - 1))
        row = max(0, min(row, self.ROWS - 1))
        return col, row

    def zone_for(self, x: float, y: float) -> PitchZone:
        """Map pitch coordinates (metres) to the corresponding PitchZone."""
        col, row = self._xy_to_col_row(x, y)
        return self._zones[(col, row)]

    def xt_at(self, x: float, y: float) -> float:
        """Return the xT value at pitch coordinates (x, y) in metres."""
        return self.zone_for(x, y).xt_value

    # ------------------------------------------------------------------
    # Touch / pass evaluation
    # ------------------------------------------------------------------

    def evaluate_touch(self, touch: TouchEvent) -> float:
        """
        Compute xT gain for a single touch/pass and cache it on the object.

        - Successful action: ``xt_gain = xt(to) - xt(from)``
        - Failed action:     ``xt_gain = -xt(from) × 0.5``  (turnover cost)

        Returns
        -------
        float
            The xT gain (can be negative).
        """
        xt_from = self.xt_at(touch.from_x, touch.from_y)

        if touch.is_successful:
            xt_to = self.xt_at(touch.to_x, touch.to_y)
            gain = xt_to - xt_from
        else:
            # Turnover: lose half the threat value of the zone ceded
            gain = -xt_from * 0.5

        touch._xt_gain = gain  # type: ignore[attr-defined]
        logger.debug(
            "evaluate_touch player=%s action=%s outcome=%s gain=%.4f",
            touch.player,
            touch.action,
            touch.outcome,
            gain,
        )
        return gain

    def evaluate_sequence(self, touches: List[TouchEvent]) -> List[float]:
        """
        Evaluate xT for each touch in a sequence.

        Parameters
        ----------
        touches:
            Ordered list of ``TouchEvent`` objects (a single possession chain
            or an entire half etc.).

        Returns
        -------
        List[float]
            xT gain for each touch, in the same order.
        """
        return [self.evaluate_touch(t) for t in touches]

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def team_report(
        self,
        touches: List[TouchEvent],
        team: str,
        opponent_team: str,
    ) -> XTReport:
        """
        Generate an XTReport for *team* from a list of all game touches.

        Steps
        -----
        1. Partition touches into ``team`` and ``opponent_team`` subsets.
        2. Evaluate all touches.
        3. Accumulate total_xt (positive gains only), zone totals, player totals.
        4. Compute per-touch and dominance metrics.
        5. Return the top-5 zones and top-5 players by xT contribution.
        """
        team_touches = [t for t in touches if t.team == team]
        opp_touches = [t for t in touches if t.team == opponent_team]

        # Evaluate both sets
        self.evaluate_sequence(team_touches)
        self.evaluate_sequence(opp_touches)

        # Accumulate team stats
        zone_totals: Dict[Tuple[int, int], float] = {}
        player_totals: Dict[str, float] = {}
        total_xt = 0.0
        n_successful_passes = 0

        for touch in team_touches:
            gain = touch.xt_gain
            if gain > 0:
                total_xt += gain

                # Zone accumulation (keyed on from-zone for context)
                col, row = self._xy_to_col_row(touch.from_x, touch.from_y)
                zone_totals[(col, row)] = zone_totals.get((col, row), 0.0) + gain

                # Player accumulation
                player_totals[touch.player] = (
                    player_totals.get(touch.player, 0.0) + gain
                )

            if touch.action == "pass" and touch.is_successful:
                n_successful_passes += 1

        # Opponent xT (what they generated against us)
        total_xt_conceded = sum(t.xt_gain for t in opp_touches if t.xt_gain > 0)

        n_touches = len(team_touches)
        xt_per_touch = total_xt / n_touches if n_touches > 0 else 0.0
        dominance = total_xt / (total_xt + total_xt_conceded + 1e-9)

        # Top 5 zones by cumulative xT gained
        top_zones: List[Tuple[int, int, float]] = [
            (col, row, val)
            for (col, row), val in sorted(
                zone_totals.items(), key=lambda kv: kv[1], reverse=True
            )[:5]
        ]

        # Top 5 players by cumulative xT contribution
        top_players: List[Tuple[str, float]] = sorted(
            player_totals.items(), key=lambda kv: kv[1], reverse=True
        )[:5]

        report = XTReport(
            team=team,
            total_xt=round(total_xt, 6),
            total_xt_conceded=round(total_xt_conceded, 6),
            xt_per_touch=round(xt_per_touch, 6),
            xt_dominance=round(dominance, 6),
            top_zones=top_zones,
            top_players=top_players,
            n_touches=n_touches,
            n_successful_passes=n_successful_passes,
        )
        logger.info("Generated %s", report)
        return report

    def dominance_map(self, touches: List[TouchEvent]) -> Dict[Tuple[int, int], float]:
        """
        For every grid zone compute a dominance score based on the xT that
        each team generated *from* that zone.

        Formula
        -------
        ``dominance = (home_xt - away_xt) / (home_xt + away_xt + 1e-9)``

        Returns a dict ``{(col, row): score}`` where score ∈ [−1, +1].
        +1 means the home team completely dominates that zone; −1 the away.

        The "home" team is the team of the first touch in the list that has a
        non-empty team field.  If the list contains touches from exactly two
        distinct teams, the first team encountered is treated as home.
        """
        # Evaluate every touch
        self.evaluate_sequence(touches)

        # Identify the two teams (first encountered = home)
        teams_seen: List[str] = []
        for t in touches:
            if t.team and t.team not in teams_seen:
                teams_seen.append(t.team)
            if len(teams_seen) == 2:
                break

        if not teams_seen:
            return {}

        home_team = teams_seen[0]
        away_team = teams_seen[1] if len(teams_seen) > 1 else None

        home_zone: Dict[Tuple[int, int], float] = {}
        away_zone: Dict[Tuple[int, int], float] = {}

        for touch in touches:
            gain = touch.xt_gain
            if gain <= 0:
                continue
            col, row = self._xy_to_col_row(touch.from_x, touch.from_y)
            key = (col, row)
            if touch.team == home_team:
                home_zone[key] = home_zone.get(key, 0.0) + gain
            elif touch.team == away_team:
                away_zone[key] = away_zone.get(key, 0.0) + gain

        result: Dict[Tuple[int, int], float] = {}
        all_keys = set(home_zone) | set(away_zone)
        for key in all_keys:
            h = home_zone.get(key, 0.0)
            a = away_zone.get(key, 0.0)
            result[key] = (h - a) / (h + a + 1e-9)

        return result

    def xt_surface(self) -> List[List[float]]:
        """Return the full 12×16 xT grid as a 2D list (rows × cols)."""
        return [list(row) for row in self._grid]

    def describe(self) -> str:
        """
        Return an ASCII heatmap of the xT grid.

        Each cell is rendered as one of five characters:
        ``·  ░  ▒  ▓  █`` (increasing threat).  Rows are printed top-to-bottom
        (row 11 first) so the display matches a pitch viewed from the side with
        attack to the right.
        """
        max_val = max(v for row in self._grid for v in row)

        def _char(v: float) -> str:
            ratio = v / max_val if max_val > 0 else 0.0
            if ratio < 0.05:
                return "·"
            if ratio < 0.20:
                return "░"
            if ratio < 0.45:
                return "▒"
            if ratio < 0.75:
                return "▓"
            return "█"

        lines: List[str] = []
        lines.append(
            f"xT Surface ({self.ROWS}×{self.COLS})  "
            f"[· < 5%  ░ 5–20%  ▒ 20–45%  ▓ 45–75%  █ >75% of max]"
        )
        lines.append("  " + "".join(f"{c:2d}" for c in range(self.COLS)) + "  col")
        for row_idx in range(self.ROWS - 1, -1, -1):
            row_data = self._grid[row_idx]
            cells = "".join(_char(v) + " " for v in row_data)
            lines.append(f"{row_idx:2d} {cells}")
        lines.append("   " + "-" * (self.COLS * 2) + " →attack")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def from_dicts(touch_dicts: List[dict]) -> List[TouchEvent]:
    """
    Convert a list of dicts to ``TouchEvent`` objects.

    Required keys
    -------------
    ``player``, ``team``, ``from_x``, ``from_y``, ``to_x``, ``to_y``,
    ``action``, ``minute``, ``outcome``

    Optional keys
    -------------
    ``timestamp`` (float, default 0.0)

    Raises
    ------
    KeyError
        If a required key is missing from any dict.
    ValueError
        If a coordinate cannot be cast to float or ``minute`` to int.
    """
    events: List[TouchEvent] = []
    required = (
        "player",
        "team",
        "from_x",
        "from_y",
        "to_x",
        "to_y",
        "action",
        "minute",
        "outcome",
    )
    for idx, d in enumerate(touch_dicts):
        missing = [k for k in required if k not in d]
        if missing:
            raise KeyError(
                f"Touch dict at index {idx} is missing required keys: {missing}"
            )
        try:
            event = TouchEvent(
                player=str(d["player"]),
                team=str(d["team"]),
                from_x=float(d["from_x"]),
                from_y=float(d["from_y"]),
                to_x=float(d["to_x"]),
                to_y=float(d["to_y"]),
                action=str(d["action"]),
                minute=int(d["minute"]),
                outcome=str(d["outcome"]),
                timestamp=float(d.get("timestamp", 0.0)),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid value in touch dict at index {idx}: {exc}"
            ) from exc
        events.append(event)

    logger.debug("from_dicts: converted %d dicts to TouchEvent objects", len(events))
    return events


def compute_team_xt(
    touch_dicts: List[dict],
    team: str,
    opponent_team: str,
) -> XTReport:
    """
    One-shot convenience wrapper.

    Converts *touch_dicts* to ``TouchEvent`` objects, runs ``XTModel``
    with the default grid, and returns an ``XTReport`` for *team*.

    Parameters
    ----------
    touch_dicts:
        Raw event dicts (see ``from_dicts`` for required keys).
    team:
        Name of the team to report on.
    opponent_team:
        Name of the opposing team.

    Returns
    -------
    XTReport
    """
    model = XTModel()
    touches = from_dicts(touch_dicts)
    return model.team_report(touches, team=team, opponent_team=opponent_team)

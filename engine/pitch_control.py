"""
Pitch Control Model — Spearman (2018) Influence-Field Approach
==============================================================

For every point on the pitch, computes the probability that the home or away
team would control the ball if it reached that point.  The result is a 2-D
control surface that reveals territorial dominance *before* shots are taken.

Algorithm
---------
1. For each player, compute the time-to-intercept ``t_i`` to each grid cell:

       # player reaches (tx, ty) from (px + vx*r, py + vy*r) after reaction r
       t_i = r + sqrt((tx - px - vx*r)^2 + (ty - py - vy*r)^2) / max_speed

2. Convert time to an influence value via a Gaussian kernel:

       influence_i = exp(-t_i^2 / (2 * sigma_t^2))    sigma_t = 0.45 s

3. Aggregate per team:

       pc_home(cell) = sum_home(influence) / (sum_home + sum_away + eps)

4. Integrate with the xT grid (from expected_threat.py) to produce
   a *Dangerous Control Map*: high xT zones weighted by pitch control.

Usage
-----
    from engine.pitch_control import run_pitch_control

    result = run_pitch_control(
        home_positions=[
            {"player_id": "H1", "x": 60.0, "y": 34.0},
            {"player_id": "H2", "x": 75.0, "y": 20.0},
        ],
        away_positions=[
            {"player_id": "A1", "x": 45.0, "y": 34.0},
        ],
        ball_x=55.0,
        ball_y=34.0,
    )
    print(result.home_territory_pct, result.dangerous_home_pct)
    print(PitchControlModel().describe(result))
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PITCH_LENGTH = 105.0   # metres (x axis: 0 = own goal line, 105 = opp goal line)
_PITCH_WIDTH = 68.0     # metres (y axis)

_REACTION_TIME = 0.7    # seconds — player reaction before accelerating
_SIGMA_T = 0.45         # seconds — Gaussian kernel spread on time
_DEFAULT_MAX_SPEED = 8.0  # m/s typical sprint speed
_EPSILON = 1e-9

# Final-third boundaries
_HOME_FINAL_THIRD_MIN_X = 70.0   # home team attacks toward x=105
_AWAY_FINAL_THIRD_MAX_X = 35.0   # away team attacks toward x=0


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class PlayerState:
    """Instantaneous kinematic state of one player."""

    player_id: str
    team: str           # "home" | "away"
    x: float            # metres along pitch (0–105)
    y: float            # metres across pitch (0–68)
    vx: float = 0.0     # velocity component in x (m/s)
    vy: float = 0.0     # velocity component in y (m/s)
    max_speed: float = _DEFAULT_MAX_SPEED  # sprint speed (m/s)


@dataclass
class PitchControlFrame:
    """A single snapshot of player positions and ball location."""

    timestamp: float
    home_players: List[PlayerState]
    away_players: List[PlayerState]
    ball_x: float
    ball_y: float


@dataclass
class PitchControlResult:
    """Full pitch-control surface for one frame."""

    frame: PitchControlFrame
    grid_cols: int
    grid_rows: int
    # [row][col] probability that home team controls this cell (0–1)
    home_control: List[List[float]] = field(default_factory=list)
    # [row][col] = 1 - home_control[row][col]
    away_control: List[List[float]] = field(default_factory=list)
    home_territory_pct: float = 0.0   # fraction of cells where home_control > 0.5
    away_territory_pct: float = 0.0
    dangerous_home_pct: float = 0.0   # home-controlled cells in home final third
    dangerous_away_pct: float = 0.0   # away-controlled cells in away final third


@dataclass
class DangerousControlSummary:
    """xT-weighted pitch control per team."""

    home_xt_control: float   # sum(pc_home × xt) over all cells
    away_xt_control: float
    home_dominance: float    # (home - away) / (home + away + eps)
    interpretation: str


# ---------------------------------------------------------------------------
# Core engine
# ---------------------------------------------------------------------------


class PitchControlModel:
    """
    Computes the Spearman pitch control surface from a PitchControlFrame.

    Parameters
    ----------
    grid_cols : int
        Number of columns in the control grid (default 32 ≈ 3.28 m/cell).
    grid_rows : int
        Number of rows in the control grid (default 20 ≈ 3.40 m/cell).
    reaction_time : float
        Seconds of reaction before a player starts moving (default 0.7).
    sigma_t : float
        Width of the Gaussian time kernel in seconds (default 0.45).
    """

    def __init__(
        self,
        grid_cols: int = 32,
        grid_rows: int = 20,
        reaction_time: float = _REACTION_TIME,
        sigma_t: float = _SIGMA_T,
    ) -> None:
        self._cols = grid_cols
        self._rows = grid_rows
        self._reaction = reaction_time
        self._sigma_t = sigma_t
        self._two_sigma_sq = 2.0 * sigma_t * sigma_t

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def compute(self, frame: PitchControlFrame) -> PitchControlResult:
        """Compute pitch control for every grid cell in *frame*."""
        home_ctrl: List[List[float]] = []
        away_ctrl: List[List[float]] = []

        for row in range(self._rows):
            home_row: List[float] = []
            away_row: List[float] = []
            for col in range(self._cols):
                tx, ty = self._cell_centre(col, row)
                pc_h, pc_a = self._cell_control(frame, tx, ty)
                home_row.append(pc_h)
                away_row.append(pc_a)
            home_ctrl.append(home_row)
            away_ctrl.append(away_row)

        total_cells = self._rows * self._cols
        home_dominant = sum(
            1
            for r in range(self._rows)
            for c in range(self._cols)
            if home_ctrl[r][c] > 0.5
        )
        away_dominant = total_cells - home_dominant

        # Dangerous cells
        home_final_third_cells = [
            (r, c)
            for r in range(self._rows)
            for c in range(self._cols)
            if self._cell_centre(c, r)[0] >= _HOME_FINAL_THIRD_MIN_X
        ]
        away_final_third_cells = [
            (r, c)
            for r in range(self._rows)
            for c in range(self._cols)
            if self._cell_centre(c, r)[0] <= _AWAY_FINAL_THIRD_MAX_X
        ]

        n_home_final = len(home_final_third_cells) or 1
        n_away_final = len(away_final_third_cells) or 1

        home_dangerous = sum(
            1 for r, c in home_final_third_cells if home_ctrl[r][c] > 0.5
        )
        away_dangerous = sum(
            1 for r, c in away_final_third_cells if away_ctrl[r][c] > 0.5
        )

        result = PitchControlResult(
            frame=frame,
            grid_cols=self._cols,
            grid_rows=self._rows,
            home_control=home_ctrl,
            away_control=away_ctrl,
            home_territory_pct=home_dominant / total_cells,
            away_territory_pct=away_dominant / total_cells,
            dangerous_home_pct=home_dangerous / n_home_final,
            dangerous_away_pct=away_dangerous / n_away_final,
        )
        logger.debug(
            "PitchControl: home_territory=%.1f%% dangerous_home=%.1f%%",
            result.home_territory_pct * 100,
            result.dangerous_home_pct * 100,
        )
        return result

    def dominance_map(self, result: PitchControlResult) -> List[List[float]]:
        """
        Returns a [row][col] map in [-1, +1].
        Positive values = home dominance; negative = away dominance.
        """
        dom: List[List[float]] = []
        for row in range(result.grid_rows):
            dom_row: List[float] = []
            for col in range(result.grid_cols):
                pc_h = result.home_control[row][col]
                dom_row.append(2.0 * pc_h - 1.0)
            dom.append(dom_row)
        return dom

    def dangerous_zones(self, result: PitchControlResult) -> Dict[str, float]:
        """
        Returns percentage of the attacking final-third controlled by each team.
        """
        return {
            "home_attack": result.dangerous_home_pct,
            "away_attack": result.dangerous_away_pct,
        }

    def describe(self, result: PitchControlResult) -> str:
        """
        ASCII visualisation of the pitch control surface.

        Characters: H (>0.8), h (>0.6), . (0.4–0.6), a (<0.4), A (<0.2)
        Rows from top (y=68) to bottom (y=0); columns left (x=0) to right (x=105).
        """
        lines = [
            f"Pitch Control — {result.grid_cols}×{result.grid_rows} grid",
            f"Home territory: {result.home_territory_pct*100:.1f}%  "
            f"Away territory: {result.away_territory_pct*100:.1f}%",
            f"Dangerous (home final 3rd): {result.dangerous_home_pct*100:.1f}%  "
            f"Dangerous (away final 3rd): {result.dangerous_away_pct*100:.1f}%",
            "+" + "-" * result.grid_cols + "+",
        ]
        for row_idx in range(result.grid_rows - 1, -1, -1):
            chars = []
            for col_idx in range(result.grid_cols):
                pc = result.home_control[row_idx][col_idx]
                if pc > 0.80:
                    chars.append("H")
                elif pc > 0.60:
                    chars.append("h")
                elif pc > 0.40:
                    chars.append(".")
                elif pc > 0.20:
                    chars.append("a")
                else:
                    chars.append("A")
            lines.append("|" + "".join(chars) + "|")
        lines.append("+" + "-" * result.grid_cols + "+")
        lines.append("Legend: H=home dominant  h=home lean  .=contested  a=away lean  A=away dominant")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _cell_centre(self, col: int, row: int) -> Tuple[float, float]:
        """Return the (x, y) coordinates of the centre of a grid cell."""
        cell_w = _PITCH_LENGTH / self._cols
        cell_h = _PITCH_WIDTH / self._rows
        tx = (col + 0.5) * cell_w
        ty = (row + 0.5) * cell_h
        return tx, ty

    def _time_to_intercept(self, player: PlayerState, tx: float, ty: float) -> float:
        """
        Estimated time for *player* to reach target (tx, ty) in seconds.

        After the reaction time ``r``, the player is at
        (px + vx*r, py + vy*r) and sprints at ``max_speed``.
        """
        rx = player.x + player.vx * self._reaction
        ry = player.y + player.vy * self._reaction
        dist = math.sqrt((tx - rx) ** 2 + (ty - ry) ** 2)
        sprint_time = dist / max(player.max_speed, 0.1)
        return self._reaction + sprint_time

    def _player_influence(self, player: PlayerState, tx: float, ty: float) -> float:
        """Gaussian influence of one player at target point (tx, ty)."""
        t_intercept = self._time_to_intercept(player, tx, ty)
        return math.exp(-(t_intercept ** 2) / self._two_sigma_sq)

    def _cell_control(
        self, frame: PitchControlFrame, tx: float, ty: float
    ) -> Tuple[float, float]:
        """Return (pc_home, pc_away) for target point (tx, ty)."""
        home_inf = sum(
            self._player_influence(p, tx, ty) for p in frame.home_players
        )
        away_inf = sum(
            self._player_influence(p, tx, ty) for p in frame.away_players
        )
        total = home_inf + away_inf + _EPSILON
        return home_inf / total, away_inf / total


# ---------------------------------------------------------------------------
# xT integration
# ---------------------------------------------------------------------------

# Default xT grid from expected_threat.py (12 rows × 16 cols, home attacking left→right)
_DEFAULT_XT_GRID: List[List[float]] = [
    [0.001, 0.002, 0.003, 0.004, 0.005, 0.007, 0.010, 0.014, 0.020, 0.030, 0.050, 0.080, 0.120, 0.180, 0.260, 0.340],
    [0.001, 0.002, 0.004, 0.005, 0.007, 0.010, 0.014, 0.019, 0.028, 0.042, 0.068, 0.105, 0.155, 0.220, 0.300, 0.380],
    [0.002, 0.003, 0.005, 0.007, 0.010, 0.014, 0.019, 0.027, 0.039, 0.058, 0.090, 0.135, 0.190, 0.260, 0.340, 0.430],
    [0.002, 0.004, 0.006, 0.009, 0.013, 0.018, 0.025, 0.035, 0.050, 0.075, 0.115, 0.165, 0.230, 0.310, 0.400, 0.490],
    [0.003, 0.005, 0.008, 0.011, 0.016, 0.022, 0.031, 0.043, 0.062, 0.093, 0.140, 0.200, 0.275, 0.365, 0.460, 0.550],
    [0.003, 0.006, 0.009, 0.013, 0.019, 0.026, 0.037, 0.052, 0.074, 0.110, 0.165, 0.235, 0.320, 0.420, 0.520, 0.610],
    [0.003, 0.006, 0.009, 0.013, 0.019, 0.026, 0.037, 0.052, 0.074, 0.110, 0.165, 0.235, 0.320, 0.420, 0.520, 0.610],
    [0.003, 0.005, 0.008, 0.011, 0.016, 0.022, 0.031, 0.043, 0.062, 0.093, 0.140, 0.200, 0.275, 0.365, 0.460, 0.550],
    [0.002, 0.004, 0.006, 0.009, 0.013, 0.018, 0.025, 0.035, 0.050, 0.075, 0.115, 0.165, 0.230, 0.310, 0.400, 0.490],
    [0.002, 0.003, 0.005, 0.007, 0.010, 0.014, 0.019, 0.027, 0.039, 0.058, 0.090, 0.135, 0.190, 0.260, 0.340, 0.430],
    [0.001, 0.002, 0.004, 0.005, 0.007, 0.010, 0.014, 0.019, 0.028, 0.042, 0.068, 0.105, 0.155, 0.220, 0.300, 0.380],
    [0.001, 0.002, 0.003, 0.004, 0.005, 0.007, 0.010, 0.014, 0.020, 0.030, 0.050, 0.080, 0.120, 0.180, 0.260, 0.340],
]
# _DEFAULT_XT_GRID[row][col]: row 0=bottom, row 11=top; col 0=own goal, col 15=opp goal


class DangerousControlMap:
    """
    Combines a PitchControlResult with the xT grid to compute expected-threat-
    weighted pitch control per team.

    The xT grid uses a 12×16 cell layout; the pitch control grid may use a
    different resolution.  Bilinear interpolation maps between the two.
    """

    def __init__(self, xt_grid: Optional[List[List[float]]] = None) -> None:
        self._xt = xt_grid if xt_grid is not None else _DEFAULT_XT_GRID
        self._xt_rows = len(self._xt)
        self._xt_cols = len(self._xt[0]) if self._xt else 16

    def compute(self, pc_result: PitchControlResult) -> DangerousControlSummary:
        """
        Returns xT-weighted control per team.

        For each pitch-control cell, bilinear-interpolate the xT value and
        accumulate ``xT × pc_home`` (home) and ``xT × pc_away`` (away).
        """
        home_total = 0.0
        away_total = 0.0

        for row in range(pc_result.grid_rows):
            for col in range(pc_result.grid_cols):
                xt_home = self._interpolate_xt(
                    col, row, pc_result.grid_cols, pc_result.grid_rows
                )
                # Away team attacks in the opposite direction: mirror the x-axis
                xt_away = self._interpolate_xt(
                    pc_result.grid_cols - 1 - col, row,
                    pc_result.grid_cols, pc_result.grid_rows,
                )
                pc_h = pc_result.home_control[row][col]
                pc_a = pc_result.away_control[row][col]
                home_total += xt_home * pc_h
                away_total += xt_away * pc_a

        dominance = (home_total - away_total) / (home_total + away_total + _EPSILON)

        if dominance > 0.20:
            interp = f"HOME dominates dangerous zones (+{dominance:.2f})"
        elif dominance < -0.20:
            interp = f"AWAY dominates dangerous zones ({dominance:.2f})"
        else:
            interp = f"Contested (dominance={dominance:+.2f})"

        return DangerousControlSummary(
            home_xt_control=home_total,
            away_xt_control=away_total,
            home_dominance=dominance,
            interpretation=interp,
        )

    def _interpolate_xt(
        self, pc_col: int, pc_row: int, pc_cols: int, pc_rows: int
    ) -> float:
        """Bilinear interpolation of xT at a pitch-control cell centre."""
        # Normalised coordinates [0, 1]
        fx = (pc_col + 0.5) / pc_cols
        fy = (pc_row + 0.5) / pc_rows

        # Map to xT grid coordinates
        xt_col_f = fx * (self._xt_cols - 1)
        xt_row_f = fy * (self._xt_rows - 1)

        c0 = int(xt_col_f)
        r0 = int(xt_row_f)
        c1 = min(c0 + 1, self._xt_cols - 1)
        r1 = min(r0 + 1, self._xt_rows - 1)

        tc = xt_col_f - c0
        tr = xt_row_f - r0

        v00 = self._xt[r0][c0]
        v01 = self._xt[r0][c1]
        v10 = self._xt[r1][c0]
        v11 = self._xt[r1][c1]

        return (
            v00 * (1 - tc) * (1 - tr)
            + v01 * tc * (1 - tr)
            + v10 * (1 - tc) * tr
            + v11 * tc * tr
        )


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------


def run_pitch_control(
    home_positions: List[Dict],
    away_positions: List[Dict],
    ball_x: float = 52.5,
    ball_y: float = 34.0,
    grid_cols: int = 32,
    grid_rows: int = 20,
    timestamp: float = 0.0,
) -> PitchControlResult:
    """
    One-shot pitch control computation.

    Parameters
    ----------
    home_positions / away_positions : list of dicts
        Each dict may contain: player_id (str), x, y (float, required),
        vx, vy (float, default 0), max_speed (float, default 8.0), team (ignored).
    ball_x, ball_y : float
        Current ball position in metres.
    grid_cols, grid_rows : int
        Control grid resolution.

    Returns
    -------
    PitchControlResult
    """

    def _parse(pos_list: List[Dict], team: str) -> List[PlayerState]:
        players = []
        for pos in pos_list:
            players.append(
                PlayerState(
                    player_id=str(pos.get("player_id", f"{team}_?")),
                    team=team,
                    x=float(pos["x"]),
                    y=float(pos["y"]),
                    vx=float(pos.get("vx", 0.0)),
                    vy=float(pos.get("vy", 0.0)),
                    max_speed=float(pos.get("max_speed", _DEFAULT_MAX_SPEED)),
                )
            )
        return players

    frame = PitchControlFrame(
        timestamp=timestamp,
        home_players=_parse(home_positions, "home"),
        away_players=_parse(away_positions, "away"),
        ball_x=ball_x,
        ball_y=ball_y,
    )
    model = PitchControlModel(grid_cols=grid_cols, grid_rows=grid_rows)
    return model.compute(frame)

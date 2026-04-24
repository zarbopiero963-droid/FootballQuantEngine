"""
luck_index.py — Luck Index computation for football seasons.

Overview
--------
Teams can over- or under-perform their underlying quality because of random
variance in finishing, goalkeeper errors, and other stochastic events. By
comparing a team's *actual* points tally to *Expected Points* (xPTS) derived
from xG, we can detect systematic luck effects that the market often fails to
price correctly.

xPTS Formula
------------
For each match we model goal outcomes as independent Poisson random variables:

    P(home scores h, away scores a) = Poisson(h; xg_home) × Poisson(a; xg_away)

where Poisson(k; λ) = exp(-λ) × λ^k / k!

The score matrix is truncated at ``max_goals`` (default 8) which captures
>99.9 % of probability mass for typical xG values (≤4.0).

From the score matrix we compute match-level Expected Points:

    xPTS_home = Σ_{h>a} 3·P(h,a) + Σ_{h=a} 1·P(h,a)
    xPTS_away = Σ_{h<a} 3·P(h,a) + Σ_{h=a} 1·P(h,a)

Luck Definition
---------------
    luck(team) = actual_pts(team) - xPTS(team)

Positive luck  → team accumulated more points than their xG merited (lucky;
                 expect future regression — fade candidate).
Negative luck  → team accumulated fewer points than their xG merited (unlucky;
                 market likely underrates them — regression-to-mean candidate).

Thresholds (per-match):
    luck_per_match < -0.5  →  regression candidate  (UNLUCKY)
    luck_per_match > +0.5  →  fade candidate         (LUCKY)

Usage Example
-------------
    from engine.luck_index import compute_luck_report

    rows = [
        {
            "home_team": "Arsenal",   "away_team": "Chelsea",
            "home_goals": 2,          "away_goals": 1,
            "home_xg":    1.4,        "away_xg":    1.9,
        },
        # ... more matches
    ]

    report = compute_luck_report(rows)
    print(report.summary())

    for team in report.regression_candidates():
        print(team.team, team.luck_per_match)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level helper
# ---------------------------------------------------------------------------


def _poisson_pmf(k: int, lam: float) -> float:
    """
    Compute the Poisson probability mass function P(X = k | λ = lam).

    Uses the log-space calculation to avoid numerical underflow for large λ.
    Returns 0.0 when lam <= 0 and k > 0, and 1.0 when lam <= 0 and k == 0.
    """
    if lam <= 0.0:
        return 1.0 if k == 0 else 0.0
    if k < 0:
        return 0.0
    # log P = k*log(λ) - λ - log(k!)
    log_pmf = k * math.log(lam) - lam - math.lgamma(k + 1)
    return math.exp(log_pmf)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class MatchRecord:
    """
    Stores raw data for a single match together with derived actual points.

    Actual points are computed automatically in ``__post_init__``:
        home win  → home_pts=3, away_pts=0
        draw      → home_pts=1, away_pts=1
        away win  → home_pts=0, away_pts=3
    """

    home_team: str
    away_team: str
    home_goals: int  # actual goals scored by home side
    away_goals: int  # actual goals scored by away side
    home_xg: float  # expected goals for home side from xG model
    away_xg: float  # expected goals for away side from xG model

    # Derived — set automatically, not passed by caller
    home_pts: int = field(init=False)
    away_pts: int = field(init=False)

    def __post_init__(self) -> None:
        if self.home_goals > self.away_goals:
            self.home_pts, self.away_pts = 3, 0
        elif self.home_goals == self.away_goals:
            self.home_pts, self.away_pts = 1, 1
        else:
            self.home_pts, self.away_pts = 0, 3


@dataclass
class TeamLuckStats:
    """Aggregated luck statistics for a single team over the analysed period."""

    team: str
    matches: int
    actual_pts: int
    xpts: float  # cumulative expected points
    luck: float  # actual_pts - xpts  (positive = lucky, negative = unlucky)
    luck_per_match: float  # luck / matches
    is_regression_candidate: bool  # luck_per_match < -0.5 → unlucky, expect improvement
    is_fade_candidate: bool  # luck_per_match > +0.5 → lucky, expect regression
    xg_for: float  # cumulative xG produced
    xg_against: float  # cumulative xG conceded
    xg_diff: float  # xg_for - xg_against


@dataclass
class LuckReport:
    """
    Season-level aggregation of luck statistics for all teams.

    Attributes
    ----------
    matches_analysed : int
        Total number of match records processed.
    team_stats : Dict[str, TeamLuckStats]
        Mapping from team name to their aggregated luck statistics.
    """

    matches_analysed: int
    team_stats: Dict[str, TeamLuckStats]

    def regression_candidates(self) -> List[TeamLuckStats]:
        """Return teams sorted by luck ascending (most unlucky first)."""
        return sorted(
            [s for s in self.team_stats.values() if s.is_regression_candidate],
            key=lambda s: s.luck,
        )

    def fade_candidates(self) -> List[TeamLuckStats]:
        """Return teams sorted by luck descending (luckiest first)."""
        return sorted(
            [s for s in self.team_stats.values() if s.is_fade_candidate],
            key=lambda s: s.luck,
            reverse=True,
        )

    def summary(self) -> str:
        """
        Return a multi-line ASCII table summarising luck statistics.

        Columns: Team | Matches | ActualPts | xPTS | Luck | L/Match | Flag
        Sorted by luck ascending (most unlucky at top).
        """
        if not self.team_stats:
            return "No team data available."

        # Sort all teams by luck ascending
        rows = sorted(self.team_stats.values(), key=lambda s: s.luck)

        # Determine column widths dynamically
        headers = ["Team", "Matches", "ActualPts", "xPTS", "Luck", "L/Match", "Flag"]

        col_data: List[List[str]] = []
        for stat in rows:
            flag = ""
            if stat.is_regression_candidate:
                flag = "UNLUCKY"
            elif stat.is_fade_candidate:
                flag = "LUCKY"
            col_data.append(
                [
                    stat.team,
                    str(stat.matches),
                    str(stat.actual_pts),
                    f"{stat.xpts:.2f}",
                    f"{stat.luck:+.2f}",
                    f"{stat.luck_per_match:+.3f}",
                    flag,
                ]
            )

        # Compute column widths as max of header and all data values
        col_widths = [len(h) for h in headers]
        for row in col_data:
            for i, cell in enumerate(row):
                col_widths[i] = max(col_widths[i], len(cell))

        def _format_row(cells: List[str], widths: List[int]) -> str:
            parts = []
            for cell, width in zip(cells, widths):
                parts.append(cell.ljust(width))
            return "| " + " | ".join(parts) + " |"

        sep = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"

        lines: List[str] = []
        title = f"Luck Index Report — {self.matches_analysed} match(es) analysed"
        lines.append(title)
        lines.append("=" * len(title))
        lines.append(sep)
        lines.append(_format_row(headers, col_widths))
        lines.append(sep)
        for row in col_data:
            lines.append(_format_row(row, col_widths))
        lines.append(sep)
        lines.append(
            "Flag key: UNLUCKY = regression-to-mean candidate | LUCKY = fade candidate"
        )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core engine
# ---------------------------------------------------------------------------


class LuckIndex:
    """
    Computes Expected Points (xPTS) from xG data using a Poisson score matrix.

    For each match:
        P(h, a) = Poisson(h; xg_home) × Poisson(a; xg_away)
        xPTS_home = Σ_{h>a} 3·P(h,a) + Σ_{h=a} 1·P(h,a)
        xPTS_away = Σ_{h<a} 3·P(h,a) + Σ_{h=a} 1·P(h,a)

    Parameters
    ----------
    max_goals : int
        Score-matrix truncation ceiling (inclusive).  Default 8 captures
        >99.9 % of probability mass for xG values typically seen in football.
    """

    def __init__(self, max_goals: int = 8) -> None:
        if max_goals < 1:
            raise ValueError(f"max_goals must be >= 1, got {max_goals}")
        self.max_goals = max_goals
        logger.debug("LuckIndex initialised with max_goals=%d", max_goals)

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def xpts_from_xg(self, xg_home: float, xg_away: float) -> Tuple[float, float]:
        """
        Return (xpts_home, xpts_away) for a single match given xG values.

        Builds an (N+1) × (N+1) score-probability matrix where N = max_goals,
        then sums the appropriate cells to derive expected points.

        Parameters
        ----------
        xg_home : float  Expected goals for the home side.
        xg_away : float  Expected goals for the away side.

        Returns
        -------
        Tuple[float, float]
            (xpts_home, xpts_away) each in [0, 3].
        """
        xg_home = max(0.0, xg_home)
        xg_away = max(0.0, xg_away)

        n = self.max_goals  # upper limit (inclusive)

        # Pre-compute Poisson PMF vectors for home and away to avoid redundant calls
        home_pmf = [_poisson_pmf(k, xg_home) for k in range(n + 1)]
        away_pmf = [_poisson_pmf(k, xg_away) for k in range(n + 1)]

        xpts_home = 0.0
        xpts_away = 0.0

        for h in range(n + 1):
            ph = home_pmf[h]
            if ph == 0.0:
                continue
            for a in range(n + 1):
                pa = away_pmf[a]
                if pa == 0.0:
                    continue
                prob = ph * pa
                if h > a:
                    xpts_home += 3.0 * prob
                elif h == a:
                    xpts_home += 1.0 * prob
                    xpts_away += 1.0 * prob
                else:  # h < a
                    xpts_away += 3.0 * prob

        logger.debug(
            "xpts_from_xg(%.3f, %.3f) -> (%.4f, %.4f)",
            xg_home,
            xg_away,
            xpts_home,
            xpts_away,
        )
        return xpts_home, xpts_away

    def analyse(self, matches: List[MatchRecord]) -> LuckReport:
        """
        Process all matches and return a LuckReport.

        Algorithm
        ---------
        1. For each match, compute xPTS for home and away teams via the
           Poisson score matrix.
        2. Accumulate per team: actual_pts, xpts, xg_for, xg_against,
           match count.
        3. Compute luck = actual_pts - xpts.
        4. Set is_regression_candidate if luck_per_match < -0.5.
        5. Set is_fade_candidate if luck_per_match > +0.5.

        Parameters
        ----------
        matches : List[MatchRecord]
            One record per match; both home and away sides are processed.

        Returns
        -------
        LuckReport
        """
        # Accumulators keyed by team name
        actual_pts_acc: Dict[str, int] = {}
        xpts_acc: Dict[str, float] = {}
        xg_for_acc: Dict[str, float] = {}
        xg_against_acc: Dict[str, float] = {}
        match_count: Dict[str, int] = {}

        def _ensure_team(name: str) -> None:
            if name not in actual_pts_acc:
                actual_pts_acc[name] = 0
                xpts_acc[name] = 0.0
                xg_for_acc[name] = 0.0
                xg_against_acc[name] = 0.0
                match_count[name] = 0

        for match in matches:
            h = match.home_team
            a = match.away_team
            _ensure_team(h)
            _ensure_team(a)

            xph, xpa = self.xpts_from_xg(match.home_xg, match.away_xg)

            # Home side
            actual_pts_acc[h] += match.home_pts
            xpts_acc[h] += xph
            xg_for_acc[h] += match.home_xg
            xg_against_acc[h] += match.away_xg
            match_count[h] += 1

            # Away side
            actual_pts_acc[a] += match.away_pts
            xpts_acc[a] += xpa
            xg_for_acc[a] += match.away_xg
            xg_against_acc[a] += match.home_xg
            match_count[a] += 1

        # Build TeamLuckStats for each team
        team_stats: Dict[str, TeamLuckStats] = {}
        for team in sorted(actual_pts_acc):
            n_matches = match_count[team]
            act = actual_pts_acc[team]
            xp = xpts_acc[team]
            luck = act - xp
            lpm = luck / n_matches if n_matches > 0 else 0.0
            xgf = xg_for_acc[team]
            xga = xg_against_acc[team]

            stats = TeamLuckStats(
                team=team,
                matches=n_matches,
                actual_pts=act,
                xpts=round(xp, 4),
                luck=round(luck, 4),
                luck_per_match=round(lpm, 4),
                is_regression_candidate=lpm < -0.5,
                is_fade_candidate=lpm > 0.5,
                xg_for=round(xgf, 4),
                xg_against=round(xga, 4),
                xg_diff=round(xgf - xga, 4),
            )
            team_stats[team] = stats
            logger.debug(
                "Team %-20s  actual=%3d  xpts=%6.2f  luck=%+6.2f  lpm=%+.3f",
                team,
                act,
                xp,
                luck,
                lpm,
            )

        report = LuckReport(
            matches_analysed=len(matches),
            team_stats=team_stats,
        )
        logger.info(
            "LuckReport: %d matches, %d teams, %d regression candidates, %d fade candidates",
            len(matches),
            len(team_stats),
            len(report.regression_candidates()),
            len(report.fade_candidates()),
        )
        return report

    def from_dicts(self, rows: List[dict]) -> LuckReport:
        """
        Convenience wrapper that accepts raw dicts and returns a LuckReport.

        Expected dict keys
        ------------------
        home_team   str   required
        away_team   str   required
        home_goals  int   required
        away_goals  int   required
        home_xg     float optional — falls back to home_goals if absent
        away_xg     float optional — falls back to away_goals if absent

        The xG fallback (using actual goals when xG is unavailable) gives a
        rough but functional approximation; results are more meaningful when
        true xG data is supplied.
        """
        records: List[MatchRecord] = []
        for i, row in enumerate(rows):
            try:
                home_goals = int(row["home_goals"])
                away_goals = int(row["away_goals"])
                home_xg = float(row.get("home_xg", home_goals))
                away_xg = float(row.get("away_xg", away_goals))
                record = MatchRecord(
                    home_team=str(row["home_team"]),
                    away_team=str(row["away_team"]),
                    home_goals=home_goals,
                    away_goals=away_goals,
                    home_xg=home_xg,
                    away_xg=away_xg,
                )
                records.append(record)
            except (KeyError, ValueError, TypeError) as exc:
                logger.warning("Skipping malformed row %d: %s — %s", i, row, exc)

        logger.debug(
            "from_dicts: parsed %d/%d rows into MatchRecords", len(records), len(rows)
        )
        return self.analyse(records)


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------


def compute_luck_report(rows: List[dict], max_goals: int = 8) -> LuckReport:
    """
    One-shot convenience function: build a LuckIndex and call from_dicts.

    Parameters
    ----------
    rows : List[dict]
        List of match dicts.  See ``LuckIndex.from_dicts`` for key spec.
    max_goals : int
        Poisson score-matrix truncation ceiling (default 8).

    Returns
    -------
    LuckReport
    """
    return LuckIndex(max_goals=max_goals).from_dicts(rows)

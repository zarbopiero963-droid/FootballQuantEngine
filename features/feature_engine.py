"""
Feature engine — team strength coefficients derived exclusively from real
finished-match data stored in the local database.

Dixon-Coles style attack/defense indices
-----------------------------------------
For each team, separately for home and away contexts:

  attack_home  = (avg goals scored at home)  / league_avg_home_goals
  defense_home = (avg goals conceded at home) / league_avg_away_goals
  attack_away  = (avg goals scored away)     / league_avg_away_goals
  defense_away = (avg goals conceded away)   / league_avg_home_goals

Expected goals for a fixture:
  λ_home = league_avg_home × attack_home(H) × defense_away(A)
  λ_away = league_avg_away × attack_away(A) × defense_home(H)

All indices default to 1.0 for teams with no data.

Usage
-----
    from features.feature_engine import FeatureEngine

    engine = FeatureEngine()
    engine.fit_from_db()                  # load all FT fixtures from DB
    lh, la = engine.lambda_goals("Inter", "Milan")
    stats   = engine.team_strength("Inter")
    profile = engine.build_features()     # full dict, all teams
"""

from __future__ import annotations

import logging
import sqlite3
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from config.constants import DATABASE_NAME

logger = logging.getLogger(__name__)

# Fallback league averages (used only when DB has no data)
_DEFAULT_HOME_AVG = 1.35
_DEFAULT_AWAY_AVG = 1.10

# Minimum matches before a team's strength estimate is trusted;
# teams below this threshold are blended toward the league average.
_MIN_MATCHES = 5


# ---------------------------------------------------------------------------
# DB loader
# ---------------------------------------------------------------------------


def load_matches(
    league_id: Optional[int] = None,
    season: Optional[int] = None,
    min_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load finished fixtures (status = 'FT') from the local SQLite database.

    Parameters
    ----------
    league_id : filter to one league (None = all leagues)
    season    : filter to one season year (None = all seasons)
    min_date  : ISO date string 'YYYY-MM-DD'; only fixtures on/after this date

    Returns
    -------
    pd.DataFrame with columns: home, away, home_goals, away_goals,
    league_id, season, match_date.  Empty DataFrame if DB is unavailable.
    """
    conditions: List[str] = [
        "status = 'FT'",
        "home_goals IS NOT NULL",
        "away_goals IS NOT NULL",
    ]
    params: List[Any] = []

    if league_id is not None:
        conditions.append("league_id = ?")
        params.append(league_id)
    if season is not None:
        conditions.append("season = ?")
        params.append(season)
    if min_date is not None:
        conditions.append("match_date >= ?")
        params.append(min_date)

    where = " AND ".join(conditions)
    sql = f"""
        SELECT home, away,
               CAST(home_goals AS INTEGER) AS home_goals,
               CAST(away_goals AS INTEGER) AS away_goals,
               league_id, season, match_date
        FROM   fixtures
        WHERE  {where}
        ORDER  BY match_date
    """
    try:
        conn = sqlite3.connect(DATABASE_NAME)
        df = pd.read_sql_query(sql, conn, params=params)
        conn.close()
        logger.info("load_matches: loaded %d finished fixtures from DB", len(df))
        return df
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "load_matches: DB read failed (%s) — returning empty DataFrame", exc
        )
        return pd.DataFrame(
            columns=[
                "home",
                "away",
                "home_goals",
                "away_goals",
                "league_id",
                "season",
                "match_date",
            ]
        )


# ---------------------------------------------------------------------------
# DataFrame-level team stats builder (backward-compatible function)
# ---------------------------------------------------------------------------


def build_team_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-team mean goals for/against from a fixtures DataFrame.

    Kept for backward compatibility with callers that already have a DataFrame.

    Returns pd.DataFrame with columns: team, goals_for, goals_against, matches.
    """
    if df.empty:
        return pd.DataFrame(columns=["team", "goals_for", "goals_against", "matches"])

    home = df[["home", "home_goals", "away_goals"]].copy()
    home.columns = ["team", "goals_for", "goals_against"]

    away = df[["away", "away_goals", "home_goals"]].copy()
    away.columns = ["team", "goals_for", "goals_against"]

    combined = pd.concat([home, away], ignore_index=True)
    stats = (
        combined.groupby("team")
        .agg(
            goals_for=("goals_for", "mean"),
            goals_against=("goals_against", "mean"),
            matches=("goals_for", "count"),
        )
        .reset_index()
    )
    return stats


# ---------------------------------------------------------------------------
# FeatureEngine
# ---------------------------------------------------------------------------


class FeatureEngine:
    """
    Builds Dixon-Coles attack/defense indices from real match data.

    All strength values are ratios relative to the league average goal rate,
    so 1.0 means exactly average. Values above 1.0 are above average.

    Attributes (after fit)
    ----------------------
    league_avg_home : float  — average home goals per game across all matches
    league_avg_away : float  — average away goals per game across all matches
    _stats          : dict   — {team: {attack_home, defense_home,
                                       attack_away, defense_away,
                                       home_matches, away_matches}}
    """

    def __init__(self) -> None:
        self.league_avg_home: float = _DEFAULT_HOME_AVG
        self.league_avg_away: float = _DEFAULT_AWAY_AVG
        self._stats: Dict[str, Dict[str, float]] = {}

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit_from_db(
        self,
        league_id: Optional[int] = None,
        season: Optional[int] = None,
        min_date: Optional[str] = None,
    ) -> "FeatureEngine":
        """
        Load finished fixtures from the DB and compute team strength indices.
        Returns self for method chaining.
        """
        df = load_matches(league_id=league_id, season=season, min_date=min_date)
        return self.fit(df)

    def fit(self, df: pd.DataFrame) -> "FeatureEngine":
        """
        Compute strength indices from a fixtures DataFrame.

        df must have columns: home, away, home_goals, away_goals.
        Returns self.
        """
        self._stats = {}

        if df.empty:
            logger.warning(
                "FeatureEngine.fit: empty DataFrame — using default strengths"
            )
            return self

        df = df.dropna(subset=["home_goals", "away_goals"]).copy()
        df["home_goals"] = df["home_goals"].astype(float)
        df["away_goals"] = df["away_goals"].astype(float)

        n = len(df)
        if n == 0:
            return self

        self.league_avg_home = float(df["home_goals"].mean())
        self.league_avg_away = float(df["away_goals"].mean())

        if self.league_avg_home < 0.01:
            self.league_avg_home = _DEFAULT_HOME_AVG
        if self.league_avg_away < 0.01:
            self.league_avg_away = _DEFAULT_AWAY_AVG

        # Accumulate per-team raw sums separated by venue context
        raw: Dict[str, Dict[str, float]] = {}

        for _, row in df.iterrows():
            home = str(row["home"])
            away = str(row["away"])
            hg = float(row["home_goals"])
            ag = float(row["away_goals"])

            if home not in raw:
                raw[home] = _empty_raw()
            if away not in raw:
                raw[away] = _empty_raw()

            raw[home]["home_scored"] += hg
            raw[home]["home_conceded"] += ag
            raw[home]["home_matches"] += 1

            raw[away]["away_scored"] += ag
            raw[away]["away_conceded"] += hg
            raw[away]["away_matches"] += 1

        # Convert raw sums → Dixon-Coles strength indices
        for team, r in raw.items():
            hm = max(r["home_matches"], 1)
            am = max(r["away_matches"], 1)

            avg_home_scored = r["home_scored"] / hm
            avg_home_conceded = r["home_conceded"] / hm
            avg_away_scored = r["away_scored"] / am
            avg_away_conceded = r["away_conceded"] / am

            # Bayesian shrinkage: blend toward 1.0 for teams with few matches
            total_matches = r["home_matches"] + r["away_matches"]
            w = min(total_matches, _MIN_MATCHES) / _MIN_MATCHES

            def _index(avg: float, baseline: float) -> float:
                raw_idx = avg / baseline if baseline > 0.0 else 1.0
                return w * raw_idx + (1.0 - w) * 1.0

            self._stats[team] = {
                "attack_home": _index(avg_home_scored, self.league_avg_home),
                "defense_home": _index(avg_home_conceded, self.league_avg_away),
                "attack_away": _index(avg_away_scored, self.league_avg_away),
                "defense_away": _index(avg_away_conceded, self.league_avg_home),
                "home_matches": r["home_matches"],
                "away_matches": r["away_matches"],
            }

        logger.info(
            "FeatureEngine.fit: %d teams fitted from %d matches "
            "(avg_home=%.2f avg_away=%.2f)",
            len(self._stats),
            n,
            self.league_avg_home,
            self.league_avg_away,
        )
        return self

    # ------------------------------------------------------------------
    # Query interface
    # ------------------------------------------------------------------

    def team_strength(self, team: str) -> Dict[str, float]:
        """
        Return the strength profile for a team, defaulting to 1.0 if unknown.

        Keys: attack_home, defense_home, attack_away, defense_away,
              home_matches, away_matches.
        """
        return self._stats.get(
            team,
            {
                "attack_home": 1.0,
                "defense_home": 1.0,
                "attack_away": 1.0,
                "defense_away": 1.0,
                "home_matches": 0,
                "away_matches": 0,
            },
        )

    def lambda_goals(self, home_team: str, away_team: str) -> Tuple[float, float]:
        """
        Compute expected goals (λ_home, λ_away) for a fixture.

          λ_home = league_avg_home × attack_home(H) × defense_away(A)
          λ_away = league_avg_away × attack_away(A) × defense_home(H)

        Both values are clamped to a minimum of 0.10 to keep Poisson valid.
        """
        h = self.team_strength(home_team)
        a = self.team_strength(away_team)

        lam_home = self.league_avg_home * h["attack_home"] * a["defense_away"]
        lam_away = self.league_avg_away * a["attack_away"] * h["defense_home"]

        return max(0.10, lam_home), max(0.10, lam_away)

    def build_features(self) -> Dict[str, Dict[str, float]]:
        """
        Return the full team-strength dictionary.

        Example::

            {
                "Inter": {
                    "attack_home": 1.35, "defense_home": 0.82,
                    "attack_away": 1.18, "defense_away": 0.91,
                    "home_matches": 19,  "away_matches": 19,
                },
                ...
            }
        """
        return dict(self._stats)

    def known_teams(self) -> List[str]:
        """Return sorted list of teams that have been fitted."""
        return sorted(self._stats.keys())

    def __len__(self) -> int:
        return len(self._stats)

    def __repr__(self) -> str:
        return (
            f"FeatureEngine(teams={len(self._stats)}, "
            f"avg_home={self.league_avg_home:.2f}, "
            f"avg_away={self.league_avg_away:.2f})"
        )


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------


def build_features(
    league_id: Optional[int] = None,
    season: Optional[int] = None,
    min_date: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
    """
    One-shot convenience: load DB → fit → return full team-strength dict.

    Equivalent to::

        FeatureEngine().fit_from_db(...).build_features()
    """
    return (
        FeatureEngine()
        .fit_from_db(
            league_id=league_id,
            season=season,
            min_date=min_date,
        )
        .build_features()
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _empty_raw() -> Dict[str, float]:
    return {
        "home_scored": 0.0,
        "home_conceded": 0.0,
        "home_matches": 0.0,
        "away_scored": 0.0,
        "away_conceded": 0.0,
        "away_matches": 0.0,
    }

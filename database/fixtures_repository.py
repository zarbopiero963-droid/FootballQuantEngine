"""
Fixtures persistence layer — CRUD, snapshots, and season download registry.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from database.db_manager import connect

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _dict_factory(cursor, row) -> Dict[str, Any]:
    return {col[0]: row[idx] for idx, col in enumerate(cursor.description)}


# ---------------------------------------------------------------------------
# Repository
# ---------------------------------------------------------------------------


class FixturesRepository:
    """Persistence layer for fixtures, snapshots, and season download registry."""

    # ------------------------------------------------------------------
    # Fixture CRUD
    # ------------------------------------------------------------------

    def save_fixture(self, fixture: Dict[str, Any]) -> None:
        """Insert or replace a fixture (legacy minimal columns — kept for back-compat)."""
        conn = connect()
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO fixtures(
                    fixture_id, league, season, home, away,
                    match_date, home_goals, away_goals, status
                ) VALUES(?,?,?,?,?,?,?,?,?)
                """,
                (
                    fixture.get("fixture_id"),
                    fixture.get("league"),
                    fixture.get("season"),
                    fixture.get("home"),
                    fixture.get("away"),
                    fixture.get("match_date"),
                    fixture.get("home_goals"),
                    fixture.get("away_goals"),
                    fixture.get("status"),
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def upsert_fixture(self, fixture: Dict[str, Any]) -> None:
        """
        Upsert a fixture with all extended columns.

        INSERT … ON CONFLICT DO UPDATE is used so we never lose data on a
        re-fetch of the same fixture_id.  updated_at is always refreshed.
        """
        now = _utcnow()
        conn = connect()
        try:
            conn.execute(
                """
                INSERT INTO fixtures(
                    fixture_id, league_id, league, season,
                    home, away, match_date,
                    home_goals, away_goals, status,
                    venue, referee, elapsed, updated_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(fixture_id) DO UPDATE SET
                    league_id  = excluded.league_id,
                    league     = excluded.league,
                    season     = excluded.season,
                    home       = excluded.home,
                    away       = excluded.away,
                    match_date = excluded.match_date,
                    home_goals = excluded.home_goals,
                    away_goals = excluded.away_goals,
                    status     = excluded.status,
                    venue      = excluded.venue,
                    referee    = excluded.referee,
                    elapsed    = excluded.elapsed,
                    updated_at = excluded.updated_at
                """,
                (
                    fixture.get("fixture_id"),
                    fixture.get("league_id"),
                    fixture.get("league"),
                    fixture.get("season"),
                    fixture.get("home"),
                    fixture.get("away"),
                    fixture.get("match_date"),
                    fixture.get("home_goals"),
                    fixture.get("away_goals"),
                    fixture.get("status"),
                    fixture.get("venue"),
                    fixture.get("referee"),
                    fixture.get("elapsed"),
                    now,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def upsert_fixtures_bulk(self, fixtures: List[Dict[str, Any]]) -> int:
        """Upsert many fixtures in one transaction. Returns count of rows processed."""
        if not fixtures:
            return 0
        now = _utcnow()
        rows = [
            (
                f.get("fixture_id"),
                f.get("league_id"),
                f.get("league"),
                f.get("season"),
                f.get("home"),
                f.get("away"),
                f.get("match_date"),
                f.get("home_goals"),
                f.get("away_goals"),
                f.get("status"),
                f.get("venue"),
                f.get("referee"),
                f.get("elapsed"),
                now,
            )
            for f in fixtures
            if f.get("fixture_id") is not None
        ]
        conn = connect()
        try:
            conn.executemany(
                """
                INSERT INTO fixtures(
                    fixture_id, league_id, league, season,
                    home, away, match_date,
                    home_goals, away_goals, status,
                    venue, referee, elapsed, updated_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(fixture_id) DO UPDATE SET
                    league_id  = excluded.league_id,
                    league     = excluded.league,
                    season     = excluded.season,
                    home       = excluded.home,
                    away       = excluded.away,
                    match_date = excluded.match_date,
                    home_goals = excluded.home_goals,
                    away_goals = excluded.away_goals,
                    status     = excluded.status,
                    venue      = excluded.venue,
                    referee    = excluded.referee,
                    elapsed    = excluded.elapsed,
                    updated_at = excluded.updated_at
                """,
                rows,
            )
            conn.commit()
            return len(rows)
        finally:
            conn.close()

    def get_fixtures_by_status(self, *statuses: str) -> List[Dict[str, Any]]:
        """Return all fixtures whose status is in the given set."""
        if not statuses:
            return []
        placeholders = ",".join("?" * len(statuses))
        conn = connect()
        try:
            conn.row_factory = _dict_factory
            cur = conn.execute(
                f"SELECT * FROM fixtures WHERE status IN ({placeholders})",
                statuses,
            )
            return cur.fetchall()
        finally:
            conn.close()

    def get_stale_live(self, yesterday: str) -> List[Dict[str, Any]]:
        """
        Return fixtures that appear LIVE in yesterday's snapshot AND still
        carry a live status in the fixtures table.

        Called at 00:01 to catch matches that finished late or overnight.
        ``yesterday`` is an ISO date string (YYYY-MM-DD).
        """
        live_statuses = ("1H", "HT", "2H", "ET", "BT", "P", "INT", "LIVE")
        ph = ",".join("?" * len(live_statuses))
        conn = connect()
        try:
            conn.row_factory = _dict_factory
            cur = conn.execute(
                f"""
                SELECT DISTINCT f.*
                FROM   fixtures f
                JOIN   snapshots s ON s.fixture_id = f.fixture_id
                WHERE  s.snapshot_date = ?
                  AND  s.status  IN ({ph})
                  AND  f.status  IN ({ph})
                """,
                (yesterday, *live_statuses, *live_statuses),
            )
            return cur.fetchall()
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Snapshot persistence
    # ------------------------------------------------------------------

    def save_snapshot(
        self,
        snapshot_date: str,
        fixture_id: int,
        status: str,
        home_goals: Optional[int],
        away_goals: Optional[int],
        home_odds: Optional[float] = None,
        draw_odds: Optional[float] = None,
        away_odds: Optional[float] = None,
    ) -> None:
        """
        Insert one daily snapshot row.

        INSERT OR IGNORE means a row already written today for the same
        fixture is not overwritten — the first snapshot of the day wins.
        """
        now = _utcnow()
        conn = connect()
        try:
            conn.execute(
                """
                INSERT OR IGNORE INTO snapshots(
                    snapshot_date, fixture_id, status,
                    home_goals, away_goals,
                    home_odds, draw_odds, away_odds,
                    created_at
                ) VALUES(?,?,?,?,?,?,?,?,?)
                """,
                (
                    snapshot_date,
                    fixture_id,
                    status,
                    home_goals,
                    away_goals,
                    home_odds,
                    draw_odds,
                    away_odds,
                    now,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def save_snapshots_bulk(
        self,
        snapshot_date: str,
        fixtures: List[Dict[str, Any]],
    ) -> int:
        """
        Insert daily snapshots for a list of fixture dicts in one transaction.

        Keys expected per dict: fixture_id, status, home_goals, away_goals.
        Odds fields (home_odds, draw_odds, away_odds) are optional.

        Returns the number of rows actually inserted (0 if all already existed).
        """
        if not fixtures:
            return 0
        now = _utcnow()
        rows = [
            (
                snapshot_date,
                f["fixture_id"],
                f.get("status", ""),
                f.get("home_goals"),
                f.get("away_goals"),
                f.get("home_odds"),
                f.get("draw_odds"),
                f.get("away_odds"),
                now,
            )
            for f in fixtures
            if f.get("fixture_id") is not None
        ]
        if not rows:
            return 0
        conn = connect()
        try:
            conn.executemany(
                """
                INSERT OR IGNORE INTO snapshots(
                    snapshot_date, fixture_id, status,
                    home_goals, away_goals,
                    home_odds, draw_odds, away_odds,
                    created_at
                ) VALUES(?,?,?,?,?,?,?,?,?)
                """,
                rows,
            )
            conn.commit()
            return conn.execute("SELECT changes()").fetchone()[0]
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Season download registry
    # ------------------------------------------------------------------

    def is_season_downloaded(self, league_id: int, season: int) -> bool:
        """Return True if this league/season has been fully downloaded."""
        conn = connect()
        try:
            row = conn.execute(
                "SELECT 1 FROM downloaded_seasons WHERE league_id=? AND season=?",
                (league_id, season),
            ).fetchone()
            return row is not None
        finally:
            conn.close()

    def mark_season_downloaded(
        self, league_id: int, season: int, n_fixtures: int
    ) -> None:
        """Register a league/season pair as fully downloaded."""
        now = _utcnow()
        conn = connect()
        try:
            conn.execute(
                """
                INSERT INTO downloaded_seasons(league_id, season, n_fixtures, downloaded_at)
                VALUES(?,?,?,?)
                ON CONFLICT(league_id, season) DO UPDATE SET
                    n_fixtures    = excluded.n_fixtures,
                    downloaded_at = excluded.downloaded_at
                """,
                (league_id, season, n_fixtures, now),
            )
            conn.commit()
        finally:
            conn.close()

    def downloaded_seasons_for_league(self, league_id: int) -> List[int]:
        """Return sorted list of already-downloaded season years for a league."""
        conn = connect()
        try:
            rows = conn.execute(
                "SELECT season FROM downloaded_seasons WHERE league_id=? ORDER BY season",
                (league_id,),
            ).fetchall()
            return [r[0] for r in rows]
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# Internal utility
# ---------------------------------------------------------------------------


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

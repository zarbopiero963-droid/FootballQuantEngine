"""
Historical Data Bootstrap
=========================
Downloads ALL available seasons for the configured league on first run,
then saves every fixture + statistics into the local SQLite database.

On subsequent runs it only fetches seasons not yet in downloaded_seasons,
so it never re-downloads what it already has.

Usage
-----
    from data.bootstrap import DataBootstrap
    bootstrap = DataBootstrap()
    bootstrap.run_full(league_id=135, progress_cb=print)
"""
from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from typing import Callable

from config.constants import DATABASE_NAME
from quant.providers.api_football_client import APIFootballClient

logger = logging.getLogger(__name__)


class DataBootstrap:

    def __init__(self, api_key: str | None = None):
        self.client = APIFootballClient(api_key=api_key)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_full(
        self,
        league_id: int,
        progress_cb: Callable[[str], None] | None = None,
        fetch_stats: bool = True,
    ) -> dict:
        """
        Download all missing seasons for league_id.

        Returns a summary dict:
            {'seasons_downloaded': int, 'fixtures_saved': int, 'stats_saved': int}
        """
        def _log(msg: str) -> None:
            logger.info(msg)
            if progress_cb:
                progress_cb(msg)

        _log(f"[Bootstrap] Checking available seasons for league {league_id}…")
        all_seasons = self.client.get_available_seasons(league_id)
        if not all_seasons:
            _log("[Bootstrap] No seasons found — check league_id and API key.")
            return {"seasons_downloaded": 0, "fixtures_saved": 0, "stats_saved": 0}

        already_done = self._get_downloaded_seasons(league_id)
        missing = [s for s in all_seasons if s not in already_done]

        _log(f"[Bootstrap] Available: {all_seasons} | Already in DB: {sorted(already_done)} | To download: {missing}")

        seasons_ok = 0
        total_fixtures = 0
        total_stats = 0

        for season in missing:
            _log(f"[Bootstrap] Downloading season {season}…")
            try:
                n_fix, n_stats = self._download_season(league_id, season, fetch_stats, _log)
                self._mark_season_done(league_id, season, n_fix)
                seasons_ok += 1
                total_fixtures += n_fix
                total_stats += n_stats
                _log(f"[Bootstrap] Season {season} done — {n_fix} fixtures, {n_stats} stats records.")
            except Exception as exc:
                _log(f"[Bootstrap] ERROR on season {season}: {exc}")
                logger.exception("Bootstrap season %s failed", season)

        _log(f"[Bootstrap] Completed. {seasons_ok} seasons, {total_fixtures} fixtures, {total_stats} stat records.")
        return {
            "seasons_downloaded": seasons_ok,
            "fixtures_saved": total_fixtures,
            "stats_saved": total_stats,
        }

    def update_current_season(
        self,
        league_id: int,
        season: int,
        progress_cb: Callable[[str], None] | None = None,
    ) -> int:
        """
        Re-fetch only the current season to pick up new results.
        Always runs (ignores downloaded_seasons cache for this season).
        Returns number of fixtures upserted.
        """
        def _log(msg: str) -> None:
            logger.info(msg)
            if progress_cb:
                progress_cb(msg)

        _log(f"[Bootstrap] Updating current season {season} for league {league_id}…")
        n_fix, _ = self._download_season(league_id, season, fetch_stats=False, log=_log)
        self._mark_season_done(league_id, season, n_fix)
        _log(f"[Bootstrap] Update done — {n_fix} fixtures refreshed.")
        return n_fix

    # ------------------------------------------------------------------
    # Internal download logic
    # ------------------------------------------------------------------

    def _download_season(
        self,
        league_id: int,
        season: int,
        fetch_stats: bool,
        log: Callable[[str], None],
    ) -> tuple[int, int]:
        fixtures = self.client.get_fixtures(league_id, season)
        log(f"  → {len(fixtures)} fixtures fetched from API")

        conn = self._connect()
        n_saved = 0
        n_stats = 0

        try:
            for fix in fixtures:
                self._upsert_fixture(conn, fix)
                n_saved += 1

            conn.commit()

            if fetch_stats:
                finished = [f for f in fixtures if f.get("status") == "FT"]
                log(f"  → Fetching stats for {len(finished)} finished fixtures…")
                for i, fix in enumerate(finished):
                    fid = fix["fixture_id"]
                    try:
                        stats = self.client.get_fixture_stats(fid)
                        self._upsert_fixture_stats(conn, stats)
                        n_stats += 1
                    except Exception as exc:
                        logger.warning("Stats failed for fixture %s: %s", fid, exc)

                    if (i + 1) % 50 == 0:
                        conn.commit()
                        log(f"  → Stats progress: {i + 1}/{len(finished)}")

                conn.commit()

        finally:
            conn.close()

        return n_saved, n_stats

    # ------------------------------------------------------------------
    # DB helpers
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(DATABASE_NAME)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _upsert_fixture(self, conn: sqlite3.Connection, f: dict) -> None:
        conn.execute(
            """
            INSERT INTO fixtures(
                fixture_id, league_id, league, season,
                home, away, match_date,
                home_goals, away_goals, ht_home, ht_away,
                status, venue, referee, elapsed, updated_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(fixture_id) DO UPDATE SET
                status     = excluded.status,
                home_goals = excluded.home_goals,
                away_goals = excluded.away_goals,
                ht_home    = excluded.ht_home,
                ht_away    = excluded.ht_away,
                elapsed    = excluded.elapsed,
                updated_at = excluded.updated_at
            """,
            (
                f.get("fixture_id"),
                f.get("league_id"),
                f.get("league"),
                f.get("season"),
                f.get("home_team"),
                f.get("away_team"),
                f.get("match_date"),
                f.get("home_goals"),
                f.get("away_goals"),
                f.get("ht_home"),
                f.get("ht_away"),
                f.get("status"),
                f.get("venue"),
                f.get("referee"),
                f.get("elapsed"),
                datetime.now(timezone.utc).isoformat(),
            ),
        )

    def _upsert_fixture_stats(self, conn: sqlite3.Connection, s: dict) -> None:
        conn.execute(
            """
            INSERT INTO fixture_stats(
                fixture_id,
                home_corners, away_corners,
                home_shots, away_shots,
                home_shots_on, away_shots_on,
                home_possession, away_possession,
                home_yellow, away_yellow,
                home_red, away_red,
                home_xg, away_xg,
                updated_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(fixture_id) DO UPDATE SET
                home_corners    = excluded.home_corners,
                away_corners    = excluded.away_corners,
                home_shots      = excluded.home_shots,
                away_shots      = excluded.away_shots,
                home_shots_on   = excluded.home_shots_on,
                away_shots_on   = excluded.away_shots_on,
                home_possession = excluded.home_possession,
                away_possession = excluded.away_possession,
                home_yellow     = excluded.home_yellow,
                away_yellow     = excluded.away_yellow,
                home_red        = excluded.home_red,
                away_red        = excluded.away_red,
                home_xg         = excluded.home_xg,
                away_xg         = excluded.away_xg,
                updated_at      = excluded.updated_at
            """,
            (
                s.get("fixture_id"),
                s.get("home_corners"), s.get("away_corners"),
                s.get("home_shots"), s.get("away_shots"),
                s.get("home_shots_on"), s.get("away_shots_on"),
                s.get("home_possession"), s.get("away_possession"),
                s.get("home_yellow"), s.get("away_yellow"),
                s.get("home_red"), s.get("away_red"),
                s.get("home_xg"), s.get("away_xg"),
                datetime.now(timezone.utc).isoformat(),
            ),
        )

    def _get_downloaded_seasons(self, league_id: int) -> set[int]:
        try:
            conn = self._connect()
            rows = conn.execute(
                "SELECT season FROM downloaded_seasons WHERE league_id = ?",
                (league_id,),
            ).fetchall()
            conn.close()
            return {r[0] for r in rows}
        except Exception:
            return set()

    def _mark_season_done(self, league_id: int, season: int, n_fixtures: int) -> None:
        try:
            conn = self._connect()
            conn.execute(
                """
                INSERT INTO downloaded_seasons(league_id, season, n_fixtures, downloaded_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(league_id, season) DO UPDATE SET
                    n_fixtures    = excluded.n_fixtures,
                    downloaded_at = excluded.downloaded_at
                """,
                (league_id, season, n_fixtures, datetime.now(timezone.utc).isoformat()),
            )
            conn.commit()
            conn.close()
        except Exception as exc:
            logger.warning("Could not mark season %s as done: %s", season, exc)

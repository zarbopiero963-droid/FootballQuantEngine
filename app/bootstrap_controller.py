"""
Bootstrap Controller
====================
Orchestrates the full data lifecycle:

  1. First run  → DataBootstrap downloads ALL historical seasons into DB
  2. Subsequent → Only missing / current seasons are refreshed
  3. Live       → LiveUpdater thread keeps in-play scores current
  4. Prediction → MultiMarketEngine generates all markets and saves to DB

The controller exposes a clean interface consumed by the main AppController
and the UI layer (progress callbacks, live callbacks).
"""
from __future__ import annotations

import logging
import os
import sqlite3
from typing import Callable

from config.constants import DATABASE_NAME
from data.bootstrap import DataBootstrap
from data.live_updater import LiveUpdater
from database.db_manager import init_db
from quant.markets.multi_market_engine import MultiMarketEngine
from quant.providers.api_football_client import APIFootballClient
from quant.providers.league_registry import name as league_name

logger = logging.getLogger(__name__)


def _persist_corner_stats(fixture_id: str | int, home_corners, away_corners) -> None:
    """Upsert a corner stats row so API-fetched data survives the next run."""
    try:
        conn = sqlite3.connect(DATABASE_NAME)
        conn.execute(
            """
            INSERT INTO fixture_stats(fixture_id, home_corners, away_corners, updated_at)
            VALUES(?, ?, ?, datetime('now'))
            ON CONFLICT(fixture_id) DO UPDATE
              SET home_corners = excluded.home_corners,
                  away_corners = excluded.away_corners,
                  updated_at   = excluded.updated_at
            """,
            (fixture_id, home_corners, away_corners),
        )
        conn.commit()
        conn.close()
    except Exception as exc:
        logger.debug("Could not persist corner stats for %s: %s", fixture_id, exc)


class BootstrapController:

    def __init__(self, api_key: str | None = None):
        self._api_key = api_key or os.getenv("API_FOOTBALL_KEY", "")
        self._bootstrap: DataBootstrap | None = None
        self._live_updater: LiveUpdater | None = None
        self._multi_engine: MultiMarketEngine | None = None
        self._live_callbacks: list[Callable[[list[dict]], None]] = []

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def initialise(self, progress_cb: Callable[[str], None] | None = None) -> None:
        """Ensure DB tables exist and create engine instances."""
        init_db()
        self._bootstrap     = DataBootstrap(api_key=self._api_key)
        self._multi_engine  = MultiMarketEngine()
        self._live_updater  = LiveUpdater(api_key=self._api_key, interval=60)
        self._live_updater.on_update(self._on_live_update)
        if progress_cb:
            progress_cb("[Init] Engine ready.")

    # ------------------------------------------------------------------
    # Full historical bootstrap (first run)
    # ------------------------------------------------------------------

    def run_bootstrap(
        self,
        league_id: int,
        progress_cb: Callable[[str], None] | None = None,
        fetch_stats: bool = True,
    ) -> dict:
        """
        Download ALL available seasons for league_id that are not yet in DB.
        Safe to call repeatedly — only fetches what is missing.
        """
        if not self._bootstrap:
            self.initialise(progress_cb)

        return self._bootstrap.run_full(  # type: ignore[union-attr]
            league_id=league_id,
            progress_cb=progress_cb,
            fetch_stats=fetch_stats,
        )

    def update_current_season(
        self,
        league_id: int,
        season: int,
        progress_cb: Callable[[str], None] | None = None,
    ) -> int:
        """Re-fetch the current season to pick up latest results."""
        if not self._bootstrap:
            self.initialise(progress_cb)
        return self._bootstrap.update_current_season(  # type: ignore[union-attr]
            league_id=league_id,
            season=season,
            progress_cb=progress_cb,
        )

    # ------------------------------------------------------------------
    # Multi-market prediction
    # ------------------------------------------------------------------

    def run_predictions(
        self,
        league_id: int,
        season: int,
        progress_cb: Callable[[str], None] | None = None,
    ) -> list[dict]:
        """
        Fetch upcoming fixtures + odds, run MultiMarketEngine,
        save predictions to DB, return the rows for the UI.
        """
        def _log(msg: str) -> None:
            logger.info(msg)
            if progress_cb:
                progress_cb(msg)

        if not self._multi_engine:
            self.initialise(progress_cb)

        client = APIFootballClient(api_key=self._api_key)
        lname = league_name(league_id, DATABASE_NAME)

        _log(f"[Predictions] Loading completed matches — {lname} season {season}…")
        completed = client.get_completed_matches(league_id, season)
        _log(f"[Predictions] {len(completed)} completed matches loaded.")

        if not completed:
            _log(f"[Predictions] No completed matches for {lname} — run Bootstrap first.")
            return []

        # Merge corner stats from local DB; fetch from API for fixtures still missing them.
        self._merge_corner_stats(completed, client=client)

        self._multi_engine.fit(completed)  # type: ignore[union-attr]

        _log(f"[Predictions] Fetching upcoming fixtures for {lname} season {season}…")
        upcoming = client.get_upcoming_matches(league_id, season)
        _log(f"[Predictions] {len(upcoming)} upcoming fixtures found in {lname}.")

        if not upcoming:
            _log(f"[Predictions] No upcoming fixtures in {lname}.")
            return []

        fixture_ids = [f["fixture_id"] for f in upcoming]
        _log(f"[Predictions] Fetching odds for {len(fixture_ids)} {lname} fixtures…")
        odds_map = client.get_prematch_odds(fixture_ids)
        _log(f"[Predictions] Odds fetched for {len(odds_map)} fixtures.")

        _log(f"[Predictions] Computing multi-market predictions for {lname}…")
        rows = self._multi_engine.predict_all(upcoming, odds_map)  # type: ignore[union-attr]
        _log(f"[Predictions] {len(rows)} prediction rows generated for {lname}.")

        saved = self._multi_engine.save_predictions(rows, DATABASE_NAME)  # type: ignore[union-attr]
        _log(f"[Predictions] {saved} rows saved to DB.")

        return rows

    # ------------------------------------------------------------------
    # Live updater
    # ------------------------------------------------------------------

    def start_live_updates(self) -> None:
        if self._live_updater and not self._live_updater.is_running:
            self._live_updater.start()
            logger.info("Live updater started.")

    def stop_live_updates(self) -> None:
        if self._live_updater:
            self._live_updater.stop()

    def on_live_update(self, callback: Callable[[list[dict]], None]) -> None:
        self._live_callbacks.append(callback)

    def live_health(self) -> dict:
        if self._live_updater:
            return self._live_updater.health()
        return {"running": False}

    def _on_live_update(self, fixtures: list[dict]) -> None:
        for cb in self._live_callbacks:
            try:
                cb(fixtures)
            except Exception as exc:
                logger.warning("Live callback error: %s", exc)

    @staticmethod
    def _merge_corner_stats(
        matches: list[dict],
        client=None,
        api_fallback_limit: int = 30,
    ) -> None:
        """Join home_corners/away_corners into each match dict.

        First reads from the local fixture_stats table.  For any FT match
        still missing corner data, falls back to the API (capped at
        api_fallback_limit calls) and persists the result so future runs
        don't need to fetch again.
        """
        # 1. Load whatever is already stored locally.
        try:
            conn = sqlite3.connect(DATABASE_NAME)
            rows = conn.execute(
                "SELECT fixture_id, home_corners, away_corners FROM fixture_stats"
            ).fetchall()
            conn.close()
        except Exception as exc:
            logger.warning("Could not load fixture_stats for corners: %s", exc)
            rows = []

        # Keep only rows where at least one value is non-null (avoid masking API data).
        stats_map: dict[str, tuple] = {
            str(r[0]): (r[1], r[2]) for r in rows if r[1] is not None or r[2] is not None
        }

        # 2. Optionally fill gaps via API (avoids silently training on zeroed data).
        if client is not None:
            missing = [
                m["fixture_id"]
                for m in matches
                if str(m.get("fixture_id", "")) not in stats_map
            ][:api_fallback_limit]

            if missing:
                logger.info(
                    "Fetching corner stats from API for %d fixtures missing local data.",
                    len(missing),
                )

            for fid in missing:
                try:
                    stats = client.get_fixture_stats(fid)
                    hc = stats.get("home_corners")
                    ac = stats.get("away_corners")
                    if hc is not None or ac is not None:
                        stats_map[str(fid)] = (hc, ac)
                        _persist_corner_stats(fid, hc, ac)
                except Exception as exc:
                    logger.debug("API corner fetch skipped for %s: %s", fid, exc)

        # 3. Inject into match dicts (setdefault preserves any caller-supplied value).
        n_filled = 0
        for m in matches:
            hc, ac = stats_map.get(str(m.get("fixture_id", "")), (None, None))
            m.setdefault("home_corners", hc)
            m.setdefault("away_corners", ac)
            if hc is not None:
                n_filled += 1

        logger.info(
            "Corner stats merged: %d/%d matches have corner data.",
            n_filled, len(matches),
        )

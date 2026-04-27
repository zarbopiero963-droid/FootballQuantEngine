"""
Live Updater
============
Background thread that polls API-Football for live fixtures every N seconds,
updates their score/status in the DB, and notifies registered callbacks.

Typical use:
    updater = LiveUpdater(api_key="...", interval=60)
    updater.on_update(lambda fixtures: print(fixtures))
    updater.start()
    ...
    updater.stop()
"""
from __future__ import annotations

import logging
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Callable

from config.constants import DATABASE_NAME
from quant.providers.api_football_client import APIFootballClient

logger = logging.getLogger(__name__)


class LiveUpdater:

    def __init__(self, api_key: str | None = None, interval: int = 60):
        self.client = APIFootballClient(api_key=api_key)
        self.interval = interval
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._callbacks: list[Callable[[list[dict]], None]] = []
        self._lock = threading.Lock()

        # Health metrics
        self.polls_total = 0
        self.polls_errors = 0
        self.last_poll_time: str | None = None
        self.last_live_count = 0

    # ------------------------------------------------------------------
    # Public control
    # ------------------------------------------------------------------

    def on_update(self, callback: Callable[[list[dict]], None]) -> None:
        """Register a callback invoked with the list of live fixtures on each poll."""
        with self._lock:
            self._callbacks.append(callback)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            logger.warning("LiveUpdater already running")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._poll_loop,
            daemon=True,
            name="live_updater",
        )
        self._thread.start()
        logger.info("LiveUpdater started (interval=%ss)", self.interval)

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=10)
        logger.info("LiveUpdater stopped")

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def health(self) -> dict:
        return {
            "running": self.is_running,
            "polls_total": self.polls_total,
            "polls_errors": self.polls_errors,
            "error_rate": (
                round(self.polls_errors / self.polls_total, 3)
                if self.polls_total
                else 0.0
            ),
            "last_poll": self.last_poll_time,
            "last_live_count": self.last_live_count,
        }

    # ------------------------------------------------------------------
    # Poll loop
    # ------------------------------------------------------------------

    def _poll_loop(self) -> None:
        while not self._stop_event.wait(timeout=self.interval):
            self._poll_once()

    def _poll_once(self) -> None:
        self.polls_total += 1
        try:
            live_fixtures = self.client.get_live_fixtures()
            self.last_live_count = len(live_fixtures)
            self.last_poll_time = datetime.now(timezone.utc).isoformat()

            if live_fixtures:
                self._save_live_to_db(live_fixtures)
                self._notify(live_fixtures)
                logger.debug("LiveUpdater: %d live fixtures updated", len(live_fixtures))

        except Exception as exc:
            self.polls_errors += 1
            logger.error("LiveUpdater poll failed: %s", exc)

    def _notify(self, fixtures: list[dict]) -> None:
        with self._lock:
            callbacks = list(self._callbacks)
        for cb in callbacks:
            try:
                cb(fixtures)
            except Exception as exc:
                logger.warning("LiveUpdater callback error: %s", exc)

    # ------------------------------------------------------------------
    # DB persistence
    # ------------------------------------------------------------------

    def _save_live_to_db(self, fixtures: list[dict]) -> None:
        try:
            conn = sqlite3.connect(DATABASE_NAME)
            conn.execute("PRAGMA journal_mode=WAL")
            now = datetime.now(timezone.utc).isoformat()

            for f in fixtures:
                conn.execute(
                    """
                    INSERT INTO fixtures(
                        fixture_id, league_id, league, season,
                        home, away, match_date,
                        home_goals, away_goals, ht_home, ht_away,
                        status, elapsed, venue, referee, updated_at
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
                        f.get("elapsed"),
                        f.get("venue"),
                        f.get("referee"),
                        now,
                    ),
                )

            conn.commit()
            conn.close()
        except Exception as exc:
            logger.error("LiveUpdater DB save failed: %s", exc)

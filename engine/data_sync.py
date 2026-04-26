"""
Daily data synchronisation engine for API-Football v3.

State machine
-------------
  NS  (Not Started)  → saved as upcoming fixture
  LIVE statuses      → saved + snapshot every poll cycle
  FT / AET / PEN     → saved as finished; final snapshot written
  PST / CANC / ABD   → status updated; no further polling

Nightly reconciliation (00:01)
-------------------------------
  Fixtures that were LIVE in yesterday's snapshot but still show a live
  status in the DB are re-fetched individually to capture the final score.

Snapshot deduplication
----------------------
  ``snapshots`` has a UNIQUE(snapshot_date, fixture_id) constraint so
  INSERT OR IGNORE prevents duplicate rows even if sync runs multiple times
  in the same calendar day.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any, List, Optional

from data.api_football_collector import ApiFootballCollector
from database.fixtures_repository import FixturesRepository

logger = logging.getLogger(__name__)

# Statuses we treat as "the match is over"
_FINISHED = frozenset({"FT", "AET", "PEN"})
# Statuses that indicate an active / in-progress match
_LIVE = frozenset({"1H", "HT", "2H", "ET", "BT", "P", "INT", "LIVE"})
# Statuses that mean the match will not take place (no further updates needed)
_DEAD = frozenset({"PST", "CANC", "ABD", "AWD", "WO"})


# ---------------------------------------------------------------------------
# Metrics dataclass (returned after each sync run for observability)
# ---------------------------------------------------------------------------


@dataclass
class SyncMetrics:
    run_date: str = ""
    finished_upserted: int = 0
    live_upserted: int = 0
    upcoming_upserted: int = 0
    stale_live_resolved: int = 0
    snapshots_written: int = 0
    errors: List[str] = field(default_factory=list)

    def log_summary(self) -> None:
        logger.info(
            "SyncMetrics [%s]: finished=%d live=%d upcoming=%d "
            "stale_resolved=%d snapshots=%d errors=%d",
            self.run_date,
            self.finished_upserted,
            self.live_upserted,
            self.upcoming_upserted,
            self.stale_live_resolved,
            self.snapshots_written,
            len(self.errors),
        )
        for err in self.errors:
            logger.warning("  sync error: %s", err)


# ---------------------------------------------------------------------------
# DailySyncJob
# ---------------------------------------------------------------------------


class DailySyncJob:
    """
    Orchestrates the full daily sync cycle:

    1. ``update_stale_live``  — resolve yesterday's dangling LIVE fixtures
    2. ``sync_finished``      — download FT/AET/PEN from yesterday's date
    3. ``sync_live``          — download all currently-live fixtures + snapshots
    4. ``sync_upcoming``      — download the next N scheduled fixtures

    Typical usage (called by the scheduler at ~00:01 and periodically):

        job = DailySyncJob()
        metrics = job.run_full_cycle()
    """

    def __init__(
        self,
        collector: Optional[ApiFootballCollector] = None,
        repo: Optional[FixturesRepository] = None,
        upcoming_count: int = 500,
    ) -> None:
        self._collector = collector or ApiFootballCollector()
        self._repo = repo or FixturesRepository()
        self._upcoming_count = upcoming_count

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_full_cycle(self, today: Optional[date] = None) -> SyncMetrics:
        """
        Run the complete sync cycle: stale resolution → finished → live →
        upcoming.  ``today`` defaults to the current UTC date.
        """
        today = today or datetime.now(timezone.utc).date()
        yesterday = today - timedelta(days=1)
        today_str = today.isoformat()
        yesterday_str = yesterday.isoformat()

        metrics = SyncMetrics(run_date=today_str)

        self._step(
            metrics, "update_stale_live", self.update_stale_live, metrics, yesterday_str
        )
        self._step(
            metrics,
            "sync_finished",
            self.sync_finished,
            metrics,
            yesterday_str,
            today_str,
        )
        self._step(metrics, "sync_live", self.sync_live, metrics, today_str)
        self._step(metrics, "sync_upcoming", self.sync_upcoming, metrics)

        metrics.log_summary()
        return metrics

    def update_stale_live(self, metrics: SyncMetrics, yesterday: str) -> None:
        """
        At 00:01 re-fetch any fixture that was LIVE in yesterday's snapshot
        but hasn't been resolved in the fixtures table yet.

        Each such fixture is fetched individually (one API call per fixture)
        so we pick up final scores for late-finishing matches.
        """
        stale = self._repo.get_stale_live(yesterday)
        if not stale:
            logger.info("update_stale_live: no stale LIVE fixtures from %s", yesterday)
            return

        logger.info(
            "update_stale_live: resolving %d stale LIVE fixtures from %s",
            len(stale),
            yesterday,
        )
        resolved = 0
        for row in stale:
            fid = row["fixture_id"]
            try:
                updated = self._collector.get_fixture_by_id(fid)
                if updated is None:
                    logger.warning(
                        "update_stale_live: fixture %d not found in API", fid
                    )
                    continue
                self._repo.upsert_fixture(updated)
                if updated["status"] in _FINISHED:
                    # Write the final snapshot for yesterday's date
                    n = self._repo.save_snapshots_bulk(yesterday, [updated])
                    metrics.snapshots_written += n
                    resolved += 1
                    logger.debug(
                        "Stale fixture %d resolved → %s (%s %d–%d %s)",
                        fid,
                        updated["status"],
                        updated.get("home", "?"),
                        updated.get("home_goals") or 0,
                        updated.get("away_goals") or 0,
                        updated.get("away", "?"),
                    )
            except Exception as exc:  # noqa: BLE001
                msg = f"stale fixture {fid}: {exc}"
                metrics.errors.append(msg)
                logger.exception("update_stale_live error — %s", msg)

        metrics.stale_live_resolved = resolved
        logger.info(
            "update_stale_live: resolved %d/%d stale fixtures", resolved, len(stale)
        )

    def sync_finished(
        self,
        metrics: SyncMetrics,
        yesterday: str,
        today: str,
    ) -> None:
        """
        Download all finished (FT/AET/PEN) fixtures for yesterday's date
        *and* today's date (to catch matches that finished after midnight).

        Each finished fixture gets a snapshot for the date it finished.
        """
        total_upserted = 0
        for date_str in (yesterday, today):
            try:
                raw = self._collector.get_fixtures_by_date(date_str)
            except Exception as exc:  # noqa: BLE001
                msg = f"sync_finished({date_str}): {exc}"
                metrics.errors.append(msg)
                logger.exception("sync_finished error — %s", msg)
                continue

            finished = [f for f in raw if f.get("status") in _FINISHED]
            dead = [f for f in raw if f.get("status") in _DEAD]

            # Upsert finished + dead matches (dead: no score update but status must be correct)
            to_save = finished + dead
            if to_save:
                n = self._repo.upsert_fixtures_bulk(to_save)
                total_upserted += n
                logger.info(
                    "sync_finished(%s): upserted %d fixtures (%d finished, %d dead)",
                    date_str,
                    n,
                    len(finished),
                    len(dead),
                )

            # Snapshot only the truly finished ones
            if finished:
                snap_n = self._repo.save_snapshots_bulk(date_str, finished)
                metrics.snapshots_written += snap_n
                logger.info(
                    "sync_finished(%s): wrote %d new snapshots", date_str, snap_n
                )

        metrics.finished_upserted = total_upserted

    def sync_live(self, metrics: SyncMetrics, today: str) -> None:
        """
        Fetch all currently live fixtures and:
        1. Upsert into fixtures table (captures elapsed time + running score)
        2. Write a today snapshot (INSERT OR IGNORE → first write of day wins)
        """
        try:
            live_fixtures = self._collector.get_live_fixtures()
        except Exception as exc:  # noqa: BLE001
            msg = f"sync_live: {exc}"
            metrics.errors.append(msg)
            logger.exception("sync_live error — %s", msg)
            return

        if not live_fixtures:
            logger.info("sync_live: no live fixtures at this moment")
            return

        n_upserted = self._repo.upsert_fixtures_bulk(live_fixtures)
        metrics.live_upserted = n_upserted

        snap_n = self._repo.save_snapshots_bulk(today, live_fixtures)
        metrics.snapshots_written += snap_n

        logger.info(
            "sync_live: %d live fixtures upserted, %d snapshots written",
            n_upserted,
            snap_n,
        )

    def sync_upcoming(self, metrics: SyncMetrics) -> None:
        """
        Fetch the next N scheduled fixtures (status NS / TBD) and upsert them.
        This keeps the fixtures table populated ahead of match days.
        """
        try:
            upcoming = self._collector.get_next_fixtures(n=self._upcoming_count)
        except Exception as exc:  # noqa: BLE001
            msg = f"sync_upcoming: {exc}"
            metrics.errors.append(msg)
            logger.exception("sync_upcoming error — %s", msg)
            return

        if not upcoming:
            logger.info("sync_upcoming: no upcoming fixtures returned")
            return

        n = self._repo.upsert_fixtures_bulk(upcoming)
        metrics.upcoming_upserted = n
        logger.info("sync_upcoming: %d upcoming fixtures upserted", n)

    # ------------------------------------------------------------------
    # Convenience: live-only poll (used by high-frequency in-play polling)
    # ------------------------------------------------------------------

    def poll_live(self) -> SyncMetrics:
        """
        Lightweight poll: only sync_live.  Used by the high-frequency scheduler
        (e.g. every 5 minutes during match hours).
        """
        today_str = datetime.now(timezone.utc).date().isoformat()
        metrics = SyncMetrics(run_date=today_str)
        self._step(metrics, "sync_live", self.sync_live, metrics, today_str)
        metrics.log_summary()
        return metrics

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------

    @staticmethod
    def _step(
        metrics: SyncMetrics,
        name: str,
        fn,
        *args: Any,
    ) -> None:
        """Execute one sync step, catching and recording any top-level exception."""
        try:
            fn(*args)
        except Exception as exc:  # noqa: BLE001
            msg = f"{name}: unexpected error: {exc}"
            metrics.errors.append(msg)
            logger.exception("DailySyncJob step '%s' failed — %s", name, exc)

"""
Job scheduler supporting both fixed-interval jobs and wall-clock daily jobs.

Job types
---------
Interval job  — runs every N seconds (e.g. live-poll every 5 minutes)
Daily job     — runs once per day at a specific UTC wall-clock time (e.g. "00:01")

Thread safety
-------------
Jobs are registered before ``run()`` is called; no lock needed on the job list.
Each job is called in the scheduler thread (single-threaded by design — the
collector already serialises API access via its token bucket).

Usage
-----
    from engine.scheduler import Scheduler

    sched = Scheduler()
    sched.add_interval_job(poll_live,  interval_seconds=300)   # every 5 min
    sched.add_daily_job(run_nightly,   time_utc="00:01")       # every night at 00:01
    sched.run()   # blocks forever
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Job descriptors
# ---------------------------------------------------------------------------


@dataclass
class _IntervalJob:
    fn: Callable
    interval: float  # seconds
    last_run: float = field(default_factory=lambda: 0.0)
    name: str = ""

    def is_due(self, now: float) -> bool:
        return now - self.last_run >= self.interval

    def mark_ran(self, now: float) -> None:
        self.last_run = now


@dataclass
class _DailyJob:
    fn: Callable
    hour: int
    minute: int
    name: str = ""
    _last_run_date: Optional[str] = field(default=None, init=False, repr=False)

    def is_due(self, now: datetime) -> bool:
        if now.hour != self.hour or now.minute != self.minute:
            return False
        today = now.date().isoformat()
        return self._last_run_date != today

    def mark_ran(self, now: datetime) -> None:
        self._last_run_date = now.date().isoformat()


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------


class Scheduler:
    """
    Mixed interval + daily-at scheduler.

    Backwards-compatible shim: ``add_job(fn)`` registers fn as an interval job
    using the ``interval_minutes`` passed to ``__init__``.
    """

    def __init__(self, interval_minutes: float = 10.0) -> None:
        self._default_interval = interval_minutes * 60.0
        self._interval_jobs: List[_IntervalJob] = []
        self._daily_jobs: List[_DailyJob] = []
        self._running = False

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def add_job(self, fn: Callable, name: str = "") -> None:
        """Legacy: register fn as an interval job at the default interval."""
        self.add_interval_job(
            fn, self._default_interval, name=name or getattr(fn, "__name__", "")
        )

    def add_interval_job(
        self,
        fn: Callable,
        interval_seconds: float,
        name: str = "",
    ) -> None:
        """Register a job that runs every ``interval_seconds`` seconds."""
        job = _IntervalJob(
            fn=fn,
            interval=interval_seconds,
            name=name or getattr(fn, "__name__", "interval_job"),
        )
        self._interval_jobs.append(job)
        logger.info(
            "Scheduler: registered interval job '%s' every %.0fs",
            job.name,
            interval_seconds,
        )

    def add_daily_job(
        self,
        fn: Callable,
        time_utc: str,
        name: str = "",
    ) -> None:
        """
        Register a job that fires once per day at ``time_utc`` (HH:MM, UTC).

        Example: ``time_utc="00:01"`` fires just after midnight every night.
        """
        try:
            hour, minute = (int(p) for p in time_utc.split(":"))
        except ValueError as exc:
            raise ValueError(f"time_utc must be 'HH:MM', got {time_utc!r}") from exc

        job = _DailyJob(
            fn=fn,
            hour=hour,
            minute=minute,
            name=name or getattr(fn, "__name__", "daily_job"),
        )
        self._daily_jobs.append(job)
        logger.info(
            "Scheduler: registered daily job '%s' at %02d:%02d UTC",
            job.name,
            hour,
            minute,
        )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        """
        Block and run all registered jobs in priority order:
        1. Daily jobs that are due (checked first — highest priority)
        2. Interval jobs that are overdue

        Sleeps in 30-second ticks to keep CPU usage negligible while still
        reacting to daily jobs within a one-minute window.
        """
        self._running = True
        logger.info(
            "Scheduler started — %d interval job(s), %d daily job(s)",
            len(self._interval_jobs),
            len(self._daily_jobs),
        )
        while self._running:
            now_ts = time.monotonic()
            now_dt = datetime.now(timezone.utc)

            # --- daily jobs ---
            for job in self._daily_jobs:
                if job.is_due(now_dt):
                    self._run_job(job.fn, job.name)
                    job.mark_ran(now_dt)

            # --- interval jobs ---
            for job in self._interval_jobs:
                if job.is_due(now_ts):
                    self._run_job(job.fn, job.name)
                    job.mark_ran(now_ts)

            time.sleep(30)

    def stop(self) -> None:
        """Signal the scheduler loop to exit after the current tick."""
        self._running = False
        logger.info("Scheduler: stop requested")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _run_job(fn: Callable, name: str) -> None:
        start = time.monotonic()
        try:
            fn()
            elapsed = time.monotonic() - start
            logger.info("Scheduler: job '%s' completed in %.2fs", name, elapsed)
        except Exception as exc:  # noqa: BLE001
            elapsed = time.monotonic() - start
            logger.exception(
                "Scheduler: job '%s' raised after %.2fs — %s", name, elapsed, exc
            )

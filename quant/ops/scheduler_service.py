from __future__ import annotations

from engine.scheduler import Scheduler
from quant.ops.monitoring import Monitoring


class SchedulerService:

    def __init__(self):
        self.scheduler = Scheduler()
        self.monitoring = Monitoring()

    def add_recurring_job(
        self,
        name: str,
        callback,
        interval_seconds: float,
        run_immediately: bool = False,
    ):
        if hasattr(self.scheduler, "add_job"):
            try:
                self.scheduler.add_job(
                    name=name,
                    callback=callback,
                    interval_seconds=interval_seconds,
                    run_immediately=run_immediately,
                )
            except TypeError:
                self.scheduler.add_job(name, callback, interval_seconds)
        else:
            raise AttributeError("Scheduler does not expose add_job")

        self.monitoring.log(
            "scheduler_job_added",
            {
                "name": name,
                "interval_seconds": interval_seconds,
                "run_immediately": run_immediately,
            },
        )

    def start(self):
        self.monitoring.log("scheduler_started", {})

        if hasattr(self.scheduler, "start_background"):
            self.scheduler.start_background()
        elif hasattr(self.scheduler, "start"):
            self.scheduler.start()

    def stop(self):
        if hasattr(self.scheduler, "shutdown"):
            self.scheduler.shutdown()
        elif hasattr(self.scheduler, "stop_background"):
            self.scheduler.stop_background()
        elif hasattr(self.scheduler, "stop"):
            self.scheduler.stop()

        self.monitoring.log("scheduler_stopped", {})

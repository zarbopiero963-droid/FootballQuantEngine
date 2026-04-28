from __future__ import annotations

from engine.job_runner import JobRunner


class CliRunner:

    def __init__(self):

        self.runner = JobRunner()

    def run_once(self):

        return self.runner.run_cycle()

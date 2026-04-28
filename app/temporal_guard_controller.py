from __future__ import annotations

from quant.temporal.temporal_guard_runner import TemporalGuardRunner


class AppTemporalGuardController:

    def __init__(self):
        self.runner = TemporalGuardRunner()

    def run(self):
        return self.runner.run()

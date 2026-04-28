from __future__ import annotations

from quant.guard.quant_guard_runner import QuantGuardRunner


class AppQuantGuard:

    def __init__(self):
        self.runner = QuantGuardRunner()

    def run(self):
        return self.runner.run()

from __future__ import annotations

from quant.guard.quant_guard import QuantEngineGuard
from quant.guard.quant_snapshot import QuantSnapshotManager
from quant.services.quant_job_runner import QuantJobRunner


class QuantGuardRunner:

    def __init__(self):
        self.runner = QuantJobRunner()
        self.guard = QuantEngineGuard()
        self.snapshot = QuantSnapshotManager()

    def run(self) -> dict:
        records = self.runner.run_cycle()

        result = self.guard.validate_records(records)
        report_path = self.guard.save_report(result)
        snapshot_path = self.snapshot.save(result.summary)

        return {
            "passed": result.passed,
            "checks": result.checks,
            "summary": result.summary,
            "report_path": report_path,
            "snapshot_path": snapshot_path,
        }

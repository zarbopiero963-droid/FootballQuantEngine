from __future__ import annotations

from quant.guard.quant_guard import QuantEngineGuard
from quant.guard.quant_guard_baseline import QuantGuardBaseline
from quant.guard.quant_guard_reporter import QuantGuardReporter
from quant.guard.quant_snapshot import QuantSnapshotManager
from quant.services.quant_job_runner import QuantJobRunner


class QuantGuardRunner:

    def __init__(self):
        self.runner = QuantJobRunner()
        self.guard = QuantEngineGuard()
        self.snapshot = QuantSnapshotManager()
        self.reporter = QuantGuardReporter()
        self.baseline = QuantGuardBaseline()

    def run(self) -> dict:
        records = self.runner.run_cycle()

        result = self.guard.validate_records(records)
        snapshot_path = self.snapshot.save(result.summary)

        baseline_before = self.baseline.load()
        baseline_compare = self.baseline.compare(result.summary, baseline_before)

        if baseline_before is None:
            baseline_path = self.baseline.save(result.summary)
        else:
            baseline_path = self.baseline.path

        payload = {
            "passed": result.passed,
            "checks": result.checks,
            "summary": result.summary,
            "baseline": baseline_compare,
        }

        report_paths = self.reporter.save_bundle(payload)

        return {
            "passed": result.passed,
            "checks": result.checks,
            "summary": result.summary,
            "baseline": baseline_compare,
            "report_path": report_paths["json"],
            "report_markdown_path": report_paths["markdown"],
            "report_html_path": report_paths["html"],
            "snapshot_path": snapshot_path,
            "baseline_path": baseline_path,
        }

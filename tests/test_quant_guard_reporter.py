import os

from quant.guard.quant_guard_reporter import QuantGuardReporter


def test_quant_guard_reporter_saves_bundle(tmp_path):
    reporter = QuantGuardReporter(str(tmp_path))

    payload = {
        "passed": True,
        "checks": [{"name": "records_not_empty", "passed": True, "value": 3}],
        "summary": {"record_count": 3, "bet_count": 1},
    }

    paths = reporter.save_bundle(payload)

    assert os.path.exists(paths["json"])
    assert os.path.exists(paths["markdown"])
    assert os.path.exists(paths["html"])

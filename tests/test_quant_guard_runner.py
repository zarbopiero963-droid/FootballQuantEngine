from quant.guard.quant_guard_runner import QuantGuardRunner


def test_quant_guard_runner_runs():
    runner = QuantGuardRunner()
    result = runner.run()

    assert isinstance(result, dict)
    assert "passed" in result
    assert "summary" in result
    assert "report_path" in result
    assert "snapshot_path" in result

from quant.temporal.temporal_guard_runner import TemporalGuardRunner


def test_temporal_guard_runner_runs():
    result = TemporalGuardRunner().run()

    assert isinstance(result, dict)
    assert "payload" in result
    assert "passed" in result

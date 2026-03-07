from engine.job_runner import JobRunner


def test_job_runner_returns_quant_results():

    runner = JobRunner()

    results = runner.run_cycle()

    assert isinstance(results, list)
    assert len(results) > 0
    assert "fixture_id" in results[0]
    assert "market" in results[0]
    assert "decision" in results[0]

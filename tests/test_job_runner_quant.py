from __future__ import annotations

from unittest.mock import patch

from engine.job_runner import JobRunner


def test_job_runner_returns_quant_results(mock_api_client):
    with patch(
        "quant.services.quant_job_runner.APIFootballClient",
        return_value=mock_api_client,
    ):
        runner = JobRunner()
        results = runner.run_cycle()

    assert isinstance(results, list)
    assert len(results) > 0
    assert "fixture_id" in results[0]
    assert "market" in results[0]
    assert "decision" in results[0]

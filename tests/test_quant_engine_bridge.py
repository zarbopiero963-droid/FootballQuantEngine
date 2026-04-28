from __future__ import annotations

from unittest.mock import patch

from app.quant_controller import AppQuantController


def test_app_quant_controller_runs(mock_api_client):
    with patch(
        "quant.services.quant_job_runner.APIFootballClient",
        return_value=mock_api_client,
    ):
        controller = AppQuantController()

    results = controller.run_quant_cycle()

    assert isinstance(results, list)
    assert len(results) > 0

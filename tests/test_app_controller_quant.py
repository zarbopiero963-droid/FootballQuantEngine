from __future__ import annotations

from unittest.mock import patch

from app.controller import AppController


def test_app_controller_quant_flow(mock_api_client, tmp_path):
    with (
        patch(
            "quant.services.quant_job_runner.APIFootballClient",
            return_value=mock_api_client,
        ),
        patch("app.controller.init_db"),
    ):
        controller = AppController(output_dir=str(tmp_path))

    results = controller.run_predictions()

    assert isinstance(results, list)
    assert len(results) > 0
    assert "home_team" in results[0]
    assert "away_team" in results[0]
    assert "confidence" in results[0]

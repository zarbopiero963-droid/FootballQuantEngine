from __future__ import annotations

from unittest.mock import patch

from quant.services.quant_job_runner import QuantJobRunner


def test_quant_records_have_expected_fields(mock_api_client):
    with patch(
        "quant.services.quant_job_runner.APIFootballClient",
        return_value=mock_api_client,
    ):
        runner = QuantJobRunner()

    results = runner.run_cycle()

    assert isinstance(results, list)
    assert len(results) > 0

    first = results[0]

    expected = {
        "fixture_id",
        "league_id",
        "league_name",
        "league",
        "home_team",
        "away_team",
        "market",
        "probability",
        "fair_odds",
        "bookmaker_odds",
        "market_edge",
        "model_edge",
        "confidence",
        "agreement",
        "decision",
    }

    assert expected.issubset(set(first.keys()))

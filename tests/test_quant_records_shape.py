from __future__ import annotations

from unittest.mock import patch

from quant.services.quant_job_runner import QuantJobRunner

_EXPECTED_FIELDS: dict[str, type] = {
    "fixture_id": str,
    "league_id": int,
    "league_name": str,
    "league": str,
    "home_team": str,
    "away_team": str,
    "market": str,
    "probability": float,
    "fair_odds": float,
    "bookmaker_odds": float,
    "market_edge": float,
    "model_edge": float,
    "confidence": float,
    "agreement": float,
    "decision": str,
}


def test_quant_records_have_expected_fields(mock_api_client):
    with patch(
        "quant.services.quant_job_runner.APIFootballClient",
        return_value=mock_api_client,
    ):
        runner = QuantJobRunner()
        results = runner.run_cycle()

    assert isinstance(results, list)
    assert len(results) > 0

    for record in results:
        for field, expected_type in _EXPECTED_FIELDS.items():
            assert field in record, f"field '{field}' missing from record {record}"
            assert isinstance(record[field], expected_type), (
                f"field '{field}' expected {expected_type.__name__}, "
                f"got {type(record[field]).__name__}"
            )

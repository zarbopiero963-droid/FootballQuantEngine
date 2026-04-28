from __future__ import annotations

from unittest.mock import patch

from quant.services.quant_job_runner import QuantJobRunner

_EXPECTED_FIELDS: dict[str, type | tuple[type, ...]] = {
    "fixture_id": str,
    "league_id": (int, float),
    "league_name": str,
    "league": str,
    "home_team": str,
    "away_team": str,
    "market": str,
    "probability": (int, float),
    "fair_odds": (int, float),
    "bookmaker_odds": (int, float),
    "market_edge": (int, float),
    "model_edge": (int, float),
    "confidence": (int, float),
    "agreement": (int, float),
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
                f"field '{field}' expected {expected_type}, "
                f"got {type(record[field]).__name__}"
            )

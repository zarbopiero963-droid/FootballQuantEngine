from __future__ import annotations

import numbers
from unittest.mock import patch

from quant.services.quant_job_runner import QuantJobRunner

# str fields must be str; numeric fields accept any Real number (int or float)
_STR_FIELDS = {
    "fixture_id",
    "league_name",
    "league",
    "home_team",
    "away_team",
    "market",
    "decision",
}
_NUMERIC_FIELDS = {
    "league_id",
    "probability",
    "fair_odds",
    "bookmaker_odds",
    "market_edge",
    "model_edge",
    "confidence",
    "agreement",
}
_EXPECTED_FIELDS = _STR_FIELDS | _NUMERIC_FIELDS


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
        for field in _EXPECTED_FIELDS:
            assert field in record, f"field '{field}' missing from record {record}"

        for field in _STR_FIELDS:
            assert isinstance(record[field], str), (
                f"field '{field}' expected str, got {type(record[field]).__name__}"
            )

        for field in _NUMERIC_FIELDS:
            assert isinstance(record[field], numbers.Real), (
                f"field '{field}' expected a numeric type, "
                f"got {type(record[field]).__name__}"
            )

import os

import pytest

if not os.getenv("API_FOOTBALL_KEY"):
    pytest.skip("API_FOOTBALL_KEY missing", allow_module_level=True)

from quant.services.quant_job_runner import QuantJobRunner


def test_quant_records_have_expected_fields():

    runner = QuantJobRunner()

    results = runner.run_cycle()

    first = results[0]

    expected = {
        "fixture_id",
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

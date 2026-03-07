import json
import os

from quant.guard.quant_guard import QuantEngineGuard


def test_quant_guard_saves_report(tmp_path):
    records = [
        {
            "fixture_id": "1",
            "home_team": "TeamA",
            "away_team": "TeamB",
            "market": "home",
            "probability": 0.55,
            "fair_odds": 1.81,
            "bookmaker_odds": 2.00,
            "market_edge": 0.05,
            "model_edge": 0.10,
            "confidence": 0.70,
            "agreement": 0.66,
            "decision": "BET",
        }
    ]

    guard = QuantEngineGuard()
    result = guard.validate_records(records)

    report_path = tmp_path / "quant_guard_report.json"
    saved_path = guard.save_report(result, str(report_path))

    assert os.path.exists(saved_path)

    with open(saved_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    assert "passed" in payload
    assert "checks" in payload
    assert "summary" in payload

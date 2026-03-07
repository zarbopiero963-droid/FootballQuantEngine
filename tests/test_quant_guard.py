from quant.guard.quant_guard import QuantEngineGuard


def test_quant_guard_accepts_valid_records():
    records = [
        {
            "fixture_id": "1",
            "home_team": "TeamA",
            "away_team": "TeamB",
            "market": "home",
            "probability": 0.58,
            "fair_odds": 1.72,
            "bookmaker_odds": 2.05,
            "market_edge": 0.06,
            "model_edge": 0.18,
            "confidence": 0.72,
            "agreement": 0.68,
            "decision": "BET",
        }
    ]

    guard = QuantEngineGuard()
    result = guard.validate_records(records)

    assert result.passed is True
    assert result.summary["record_count"] == 1
    assert result.summary["bet_count"] == 1

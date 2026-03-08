from quant.value.value_bet_engine import ValueBetEngine


def test_value_bet_engine_enriches_records():
    engine = ValueBetEngine()

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

    result = engine.run(records, bankroll=1000.0)

    assert isinstance(result, list)
    assert len(result) == 1
    assert "ev" in result[0]
    assert "suggested_stake" in result[0]
    assert "clv_potential" in result[0]

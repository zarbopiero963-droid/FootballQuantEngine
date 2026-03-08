from quant.markets.market_value_engine import MarketValueEngine


def test_market_value_engine_enriches_record():
    engine = MarketValueEngine()

    result = engine.enrich_market_record(
        {
            "fixture_id": "1",
            "home_team": "TeamA",
            "away_team": "TeamB",
            "market": "over_25",
            "probability": 0.61,
            "bookmaker_odds": 1.90,
            "opening_odds": 1.95,
            "confidence": 0.70,
            "agreement": 0.66,
        },
        bankroll=1000.0,
    )

    assert "ev" in result
    assert "decision" in result
    assert "suggested_stake" in result

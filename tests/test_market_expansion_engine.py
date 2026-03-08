from quant.markets.market_expansion_engine import MarketExpansionEngine


def test_market_expansion_engine_runs():
    rows = MarketExpansionEngine().run()

    assert isinstance(rows, list)
    assert len(rows) > 0
    assert "market" in rows[0]
    assert rows[0]["market"] in {"over_25", "under_25", "btts_yes", "btts_no"}

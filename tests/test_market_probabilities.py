from quant.markets.market_probabilities import MarketProbabilities


def test_market_probabilities_build():
    result = MarketProbabilities().build(1.6, 1.2)

    assert "over_25" in result
    assert "under_25" in result
    assert "btts_yes" in result
    assert "btts_no" in result

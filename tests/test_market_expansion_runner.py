from quant.markets.market_expansion_runner import MarketExpansionRunner


def test_market_expansion_runner_runs():
    rows = MarketExpansionRunner().run()

    assert isinstance(rows, list)
    assert len(rows) > 0

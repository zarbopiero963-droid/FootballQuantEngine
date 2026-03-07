from engine.market_comparison import MarketComparison


def test_market_comparison_runs():

    market = MarketComparison()

    implied = market.implied_probabilities({"home": 2.0, "draw": 3.2, "away": 3.8})

    total = implied["home_win"] + implied["draw"] + implied["away_win"]

    assert 0.999 <= total <= 1.001

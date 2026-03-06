from analysis.probability_markets import ProbabilityMarkets


def test_probability_markets_runs():

    markets = ProbabilityMarkets()

    over_under = markets.over_under_25(1.4, 1.2)
    btts = markets.btts(1.4, 1.2)

    assert "over_2_5" in over_under
    assert "under_2_5" in over_under
    assert "btts_yes" in btts
    assert "btts_no" in btts

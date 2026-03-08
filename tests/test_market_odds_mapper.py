from quant.markets.market_odds_mapper import MarketOddsMapper


def test_market_odds_mapper_reads_market_odds():
    mapper = MarketOddsMapper()

    odds = {
        "over_25": 1.88,
        "under_25": 1.92,
        "btts_yes": 1.75,
        "btts_no": 2.05,
    }

    assert mapper.odds_for_market(odds, "over_25") == 1.88
    assert mapper.odds_for_market(odds, "btts_no") == 2.05

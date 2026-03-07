from strategies.no_bet_filter import NoBetFilter


def test_no_bet_filter_runs():

    filt = NoBetFilter()

    decision = filt.decide(
        probability=0.57,
        edge=0.06,
        confidence=0.72,
        agreement=0.68,
    )

    assert decision in ("BET", "WATCHLIST", "NO_BET")

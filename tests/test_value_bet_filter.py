from quant.value.value_bet_filter import ValueBetFilter


def test_value_bet_filter_returns_valid_decision():
    decision = ValueBetFilter().decide(
        probability=0.58,
        odds=2.00,
        ev=0.16,
        confidence=0.72,
        agreement=0.68,
    )

    assert decision in ("BET", "WATCHLIST", "NO_BET")

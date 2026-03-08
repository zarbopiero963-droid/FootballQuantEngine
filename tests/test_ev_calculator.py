from quant.value.ev_calculator import EVCalculator


def test_ev_calculator_outputs():
    calc = EVCalculator()

    ev = calc.expected_value(0.55, 2.10)
    fair = calc.fair_odds(0.55)
    implied = calc.implied_probability(2.10)

    assert isinstance(ev, float)
    assert fair > 0
    assert 0 < implied < 1

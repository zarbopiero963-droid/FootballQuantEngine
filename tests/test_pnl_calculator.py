from quant.clv.pnl_calculator import PnLCalculator


def test_pnl_calculator_settle():
    calc = PnLCalculator()

    assert calc.settle(10, 2.00, "WIN") == 10.0
    assert calc.settle(10, 2.00, "LOSE") == -10.0
    assert calc.settle(10, 2.00, "VOID") == 0.0

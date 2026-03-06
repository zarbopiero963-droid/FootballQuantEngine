from training.backtest_engine import BacktestEngine


def test_backtest_engine_class_exists():

    engine = BacktestEngine()

    assert isinstance(engine, BacktestEngine)

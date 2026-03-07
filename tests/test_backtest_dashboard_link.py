import pytest


def test_backtest_runner_window_import():

    try:
        from ui.backtest_runner_window import BacktestRunnerWindow
    except (ModuleNotFoundError, ImportError) as exc:
        pytest.skip(f"UI dependencies not available in CI: {exc}")

    assert BacktestRunnerWindow is not None

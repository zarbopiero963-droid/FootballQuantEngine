import pandas as pd

from quant.temporal.temporal_backtest_guard import TemporalBacktestGuard


def test_temporal_backtest_guard_runs():
    df = pd.DataFrame(
        {
            "match_date": pd.date_range("2024-01-01", periods=20, freq="D"),
            "decision": ["BET"] * 20,
            "stake": [10] * 20,
            "pnl": [5, -10, 5, 5, -10] * 4,
            "result": ["WIN", "LOSE", "WIN", "WIN", "LOSE"] * 4,
            "ev": [0.05] * 20,
            "clv_abs": [0.01] * 20,
            "clv_pct": [0.02] * 20,
        }
    )

    result = TemporalBacktestGuard().run(
        df,
        train_size=8,
        test_size=4,
        step=4,
    )

    assert isinstance(result, dict)
    assert "payload" in result
    assert "json_path" in result
    assert "html_path" in result

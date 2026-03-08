import pandas as pd

from quant.temporal.time_splitter import TimeSplitter


def test_time_splitter_rolling_windows():
    df = pd.DataFrame(
        {
            "match_date": pd.date_range("2024-01-01", periods=12, freq="D"),
            "value": list(range(12)),
        }
    )

    windows = TimeSplitter().rolling_windows(
        df,
        train_size=5,
        test_size=3,
        step=2,
    )

    assert isinstance(windows, list)
    assert len(windows) > 0
    assert "train" in windows[0]
    assert "test" in windows[0]

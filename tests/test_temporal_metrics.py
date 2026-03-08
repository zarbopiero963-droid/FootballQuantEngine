import pandas as pd

from quant.temporal.temporal_metrics import TemporalMetrics


def test_temporal_metrics_build():
    df = pd.DataFrame(
        [
            {
                "decision": "BET",
                "stake": 10,
                "pnl": 5,
                "result": "WIN",
                "ev": 0.10,
                "clv_abs": 0.02,
                "clv_pct": 0.05,
            },
            {
                "decision": "BET",
                "stake": 10,
                "pnl": -10,
                "result": "LOSE",
                "ev": 0.03,
                "clv_abs": -0.01,
                "clv_pct": -0.02,
            },
        ]
    )

    result = TemporalMetrics().build(df)

    assert result["bet_count"] == 2
    assert "roi" in result
    assert "avg_ev" in result

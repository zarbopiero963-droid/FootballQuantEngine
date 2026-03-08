from quant.temporal.temporal_guard import TemporalGuard


def test_temporal_guard_evaluate():
    result = TemporalGuard().evaluate(
        {
            "rows": 10,
            "bet_count": 3,
            "roi": 0.02,
            "yield": 0.02,
            "hit_rate": 0.50,
            "avg_ev": 0.04,
        }
    )

    assert "passed" in result
    assert "checks" in result

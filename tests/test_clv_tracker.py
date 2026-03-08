from quant.clv.clv_tracker import CLVTracker


def test_clv_tracker_outputs():
    tracker = CLVTracker()

    result = tracker.evaluate(2.10, 1.95)

    assert "clv_abs" in result
    assert "clv_pct" in result

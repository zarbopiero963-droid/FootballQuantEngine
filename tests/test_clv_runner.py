from quant.clv.clv_runner import CLVRunner


def test_clv_runner_runs():
    result = CLVRunner().run()

    assert isinstance(result, dict)
    assert "bets" in result
    assert "summary" in result
    assert "csv_path" in result

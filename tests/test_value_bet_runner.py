from quant.value.value_bet_runner import ValueBetRunner


def test_value_bet_runner_runs():
    result = ValueBetRunner().run()

    assert isinstance(result, list)
    assert len(result) > 0
    assert "ev" in result[0]

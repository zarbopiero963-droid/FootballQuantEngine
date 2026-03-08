from app.value_bet_controller import AppValueBetController


def test_app_value_bet_controller_runs():
    result = AppValueBetController().run()

    assert isinstance(result, list)
    assert len(result) > 0
    assert "suggested_stake" in result[0]

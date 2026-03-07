from app.quant_guard import AppQuantGuard


def test_app_quant_guard_runs():
    app_guard = AppQuantGuard()
    result = app_guard.run()

    assert isinstance(result, dict)
    assert "passed" in result
    assert "summary" in result

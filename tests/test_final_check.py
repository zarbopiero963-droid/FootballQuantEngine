from app.final_check import run_final_check


def test_app_final_check_runs():

    result = run_final_check()

    assert isinstance(result, dict)
    assert "ok" in result
    assert "missing" in result

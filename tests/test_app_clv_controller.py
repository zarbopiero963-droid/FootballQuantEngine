from app.clv_controller import AppCLVController


def test_app_clv_controller_runs():
    result = AppCLVController().run()

    assert isinstance(result, dict)
    assert "summary" in result

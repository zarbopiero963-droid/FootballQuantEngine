from app.temporal_guard_controller import AppTemporalGuardController


def test_app_temporal_guard_controller_runs():
    result = AppTemporalGuardController().run()

    assert isinstance(result, dict)
    assert "payload" in result

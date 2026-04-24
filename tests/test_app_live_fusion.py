from app.live_fusion import AppLiveFusion


def test_app_live_fusion_runs():
    result = AppLiveFusion().run()

    assert isinstance(result, dict)
    assert "rows" in result
    assert "count" in result
    assert isinstance(result["count"], int)

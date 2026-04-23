from quant.fusion.live_fusion_runner import LiveFusionRunner


def test_live_fusion_runner_runs():
    result = LiveFusionRunner().run()

    assert isinstance(result, dict)
    assert "rows" in result
    assert "count" in result
    assert "export_path" in result
    assert isinstance(result["count"], int)

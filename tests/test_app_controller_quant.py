from app.controller import AppController


def test_app_controller_quant_flow():

    controller = AppController()

    results = controller.run_predictions()

    assert isinstance(results, list)
    assert len(results) > 0
    assert "home_team" in results[0]
    assert "away_team" in results[0]
    assert "confidence" in results[0]

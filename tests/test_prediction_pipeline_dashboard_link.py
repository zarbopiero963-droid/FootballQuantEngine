from app.controller import AppController


def test_controller_run_predictions_exists():

    controller = AppController()

    results = controller.run_predictions()

    assert results is not None

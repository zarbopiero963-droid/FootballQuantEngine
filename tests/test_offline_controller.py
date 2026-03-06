from engine.offline_controller import OfflineController


def test_offline_controller_class_exists():

    controller = OfflineController()

    assert isinstance(controller, OfflineController)

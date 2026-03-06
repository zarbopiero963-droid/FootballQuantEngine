from engine.training_manager import TrainingManager


def test_training_manager_class_exists():

    manager = TrainingManager()

    assert isinstance(manager, TrainingManager)

from analysis.manual_context import ManualContextManager


def test_manual_context_manager_class_exists():

    manager = ManualContextManager()

    assert isinstance(manager, ManualContextManager)

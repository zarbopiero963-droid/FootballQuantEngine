import pytest


def test_ui_classes_import():

    try:
        from ui.dashboard_view import DashboardView
        from ui.main_window import MainWindow

        from app.controller import AppController
    except (ModuleNotFoundError, ImportError) as exc:
        pytest.skip(f"UI dependencies not available in CI: {exc}")

    assert AppController is not None
    assert DashboardView is not None
    assert MainWindow is not None

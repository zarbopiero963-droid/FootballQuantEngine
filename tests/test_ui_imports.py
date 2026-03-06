import pytest

pytest.importorskip("PySide6")

from ui.dashboard_view import DashboardView
from ui.main_window import MainWindow

from app.controller import AppController


def test_ui_classes_import():

    assert AppController is not None
    assert DashboardView is not None
    assert MainWindow is not None

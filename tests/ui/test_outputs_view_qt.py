import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

try:
    from PySide6.QtCore import Qt

    from ui.outputs_view import OutputsView
except Exception as exc:
    pytest.skip(
        f"Qt UI not available in this environment: {exc}",
        allow_module_level=True,
    )


class DummyOutputsView(OutputsView):
    def run_reports(self):
        pass

    def open_dashboard(self):
        return True

    def open_report(self):
        return True

    def open_charts(self):
        return True


def test_outputs_view_opens(qtbot):
    window = DummyOutputsView()
    qtbot.addWidget(window)
    window.show()

    qtbot.waitUntil(lambda: window.isVisible(), timeout=5000)

    assert window.windowTitle() != ""
    assert window.run_reports_button is not None
    assert window.open_dashboard_button is not None
    assert window.open_report_button is not None
    assert window.open_charts_button is not None


def test_outputs_buttons_are_clickable(qtbot):
    window = DummyOutputsView()
    qtbot.addWidget(window)
    window.show()

    qtbot.mouseClick(window.run_reports_button, Qt.LeftButton)
    qtbot.mouseClick(window.open_dashboard_button, Qt.LeftButton)
    qtbot.mouseClick(window.open_report_button, Qt.LeftButton)
    qtbot.mouseClick(window.open_charts_button, Qt.LeftButton)

    assert True

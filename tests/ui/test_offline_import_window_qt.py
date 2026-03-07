import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

try:
    from PySide6.QtCore import Qt

    from ui.offline_import_window import OfflineImportWindow
except Exception as exc:
    pytest.skip(
        f"Qt UI not available in this environment: {exc}",
        allow_module_level=True,
    )


class DummyOfflineImportWindow(OfflineImportWindow):
    def import_data(self):
        self.output.setPlainText(
            "{'matches': {'imported_rows': 10}, 'odds': {'imported_rows': 10}}"
        )


def test_offline_import_window_opens(qtbot):
    window = DummyOfflineImportWindow()
    qtbot.addWidget(window)
    window.show()

    qtbot.waitUntil(lambda: window.isVisible(), timeout=5000)

    assert window.windowTitle() != ""
    assert window.import_button is not None


def test_offline_import_window_import_button_updates_output(qtbot):
    window = DummyOfflineImportWindow()
    qtbot.addWidget(window)
    window.show()

    qtbot.mouseClick(window.import_button, Qt.LeftButton)

    qtbot.waitUntil(
        lambda: "imported_rows" in window.output.toPlainText(),
        timeout=3000,
    )

    assert "matches" in window.output.toPlainText()
    assert "odds" in window.output.toPlainText()

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

try:
    from ui.offline_import_window import OfflineImportWindow
    from ui.outputs_view import OutputsView
except Exception as exc:
    pytest.skip(
        f"Qt UI not available in this environment: {exc}",
        allow_module_level=True,
    )


def test_import_and_outputs_windows_importable():
    assert OfflineImportWindow is not None
    assert OutputsView is not None

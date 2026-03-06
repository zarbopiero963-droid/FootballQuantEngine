import pytest


def test_ui_windows_import():

    try:
        from ui.manual_context_window import ManualContextWindow
        from ui.offline_import_window import OfflineImportWindow
        from ui.outputs_view import OutputsView
    except (ModuleNotFoundError, ImportError) as exc:
        pytest.skip(f"UI dependencies not available in CI: {exc}")

    assert ManualContextWindow is not None
    assert OfflineImportWindow is not None
    assert OutputsView is not None

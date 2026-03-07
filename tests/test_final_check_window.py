import pytest


def test_final_check_window_import():

    try:
        from ui.final_check_window import FinalCheckWindow
    except (ModuleNotFoundError, ImportError) as exc:
        pytest.skip(f"UI dependencies not available in CI: {exc}")

    assert FinalCheckWindow is not None

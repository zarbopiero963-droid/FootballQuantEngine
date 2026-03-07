import pytest


def test_project_summary_window_import():

    try:
        from ui.project_summary_window import ProjectSummaryWindow
    except (ModuleNotFoundError, ImportError) as exc:
        pytest.skip(f"UI dependencies not available in CI: {exc}")

    assert ProjectSummaryWindow is not None

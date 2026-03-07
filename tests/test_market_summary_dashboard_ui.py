import pytest


def test_dashboard_market_summary_ui_import():

    try:
        from ui.dashboard_view import DashboardView
    except (ModuleNotFoundError, ImportError) as exc:
        pytest.skip(f"UI dependencies not available in CI: {exc}")

    assert DashboardView is not None

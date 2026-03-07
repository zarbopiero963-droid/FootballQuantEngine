import pytest


def test_dashboard_view_import():

    try:
        from ui.dashboard_view import DashboardCard, DashboardView
    except (ModuleNotFoundError, ImportError) as exc:
        pytest.skip(f"UI dependencies not available in CI: {exc}")

    assert DashboardView is not None
    assert DashboardCard is not None

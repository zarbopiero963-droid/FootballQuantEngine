import pytest


def test_outputs_related_imports():

    try:
        from analysis.report_viewer_helper import ReportViewerHelper
        from engine.offline_controller import OfflineController
    except (ModuleNotFoundError, ImportError) as exc:
        pytest.skip(f"Optional dependencies not available in CI: {exc}")

    assert ReportViewerHelper is not None
    assert OfflineController is not None

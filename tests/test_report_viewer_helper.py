from analytics.report_viewer_helper import ReportViewerHelper


def test_report_viewer_helper_class_exists():

    helper = ReportViewerHelper()

    assert isinstance(helper, ReportViewerHelper)

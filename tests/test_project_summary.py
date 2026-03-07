from app.project_summary import get_project_summary


def test_project_summary_runs():

    summary = get_project_summary()

    assert isinstance(summary, dict)
    assert "name" in summary
    assert "version" in summary
    assert "core_modules" in summary

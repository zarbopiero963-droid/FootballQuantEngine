from quant.guard.quant_guard_runner import QuantGuardRunner


def test_quant_guard_runner_reporting_fields():
    result = QuantGuardRunner().run()

    assert "report_markdown_path" in result
    assert "report_html_path" in result
    assert "baseline" in result

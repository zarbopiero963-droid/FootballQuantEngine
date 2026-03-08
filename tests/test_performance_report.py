from quant.clv.performance_report import PerformanceReport


def test_performance_report_build():
    report = PerformanceReport().build(
        [
            {
                "stake": 10,
                "ev": 0.10,
                "status": "SETTLED",
                "result": "WIN",
                "pnl": 10,
                "clv_abs": 0.02,
                "clv_pct": 0.05,
            }
        ]
    )

    assert report["total_bets"] == 1
    assert "roi" in report
    assert "avg_clv_abs" in report

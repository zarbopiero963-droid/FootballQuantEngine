from quant.guard.quant_guard_baseline import QuantGuardBaseline


def test_quant_guard_baseline_compare_without_existing_file(tmp_path):
    baseline = QuantGuardBaseline(str(tmp_path / "baseline.json"))

    result = baseline.compare({"record_count": 3}, None)

    assert result["has_baseline"] is False
    assert result["degraded"] is False

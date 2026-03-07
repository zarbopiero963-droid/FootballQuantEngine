from engine.final_integrity_check import FinalIntegrityCheck


def test_final_integrity_check_runs():

    checker = FinalIntegrityCheck()

    result = checker.run()

    assert isinstance(result, dict)
    assert "ok" in result
    assert "missing" in result
    assert "checked_count" in result

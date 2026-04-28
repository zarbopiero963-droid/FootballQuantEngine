from __future__ import annotations

from engine.final_integrity_check import FinalIntegrityCheck


def run_final_check():

    checker = FinalIntegrityCheck()

    return checker.run()

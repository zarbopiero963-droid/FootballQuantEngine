"""
Unit tests for export/csv_exporter.py and export/excel_exporter.py.

No API keys required — all tests write to tmp_path only.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

openpyxl = pytest.importorskip("openpyxl")

from export.csv_exporter import CsvExporter
from export.excel_exporter import ExcelExporter

# ---------------------------------------------------------------------------
# Shared sample data
# ---------------------------------------------------------------------------

SAMPLE_BETS = [
    {
        "match_id": "1001",
        "market": "home",
        "probability": 0.55,
        "odds": 2.1,
        "ev": 0.155,
        "kelly": 0.072,
        "kelly_stake": 18.0,
        "tier": "A",
        "model_edge": 0.05,
        "market_edge": 0.03,
        "confidence": 0.65,
        "agreement": 0.72,
        "decision": "BET",
        "league": "Serie A",
        "home_team": "Juventus",
        "away_team": "Inter",
    },
    {
        "match_id": "1002",
        "market": "away",
        "probability": 0.40,
        "odds": 3.2,
        "ev": 0.08,
        "kelly": 0.04,
        "kelly_stake": 10.0,
        "tier": "B",
        "model_edge": 0.03,
        "market_edge": 0.02,
        "confidence": 0.55,
        "agreement": 0.60,
        "decision": "WATCHLIST",
        "league": "Serie A",
        "home_team": "Milan",
        "away_team": "Roma",
    },
]

# ---------------------------------------------------------------------------
# CSV tests
# ---------------------------------------------------------------------------


def test_csv_export_creates_file(tmp_path):
    """CsvExporter.export(mode='bets') returns a path that exists."""
    exporter = CsvExporter(output_dir=str(tmp_path))
    result = exporter.export(SAMPLE_BETS, mode="bets")
    assert Path(result).exists(), f"Expected file at {result} to exist"


def test_csv_export_full_contains_all_rows(tmp_path):
    """mode='full' CSV has at least 2 data rows (excluding comment/header lines)."""
    exporter = CsvExporter(output_dir=str(tmp_path))
    result = exporter.export(SAMPLE_BETS, mode="full")
    path = Path(result)
    assert path.exists()

    # Count non-comment, non-header rows
    data_rows = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("#") or not stripped:
                continue
            data_rows.append(stripped)

    # First non-comment line is the header; subsequent are data rows
    assert len(data_rows) >= 3, (
        f"Expected header + at least 2 data rows, got {len(data_rows)} non-comment lines"
    )


def test_csv_export_bets_only_has_bet_rows(tmp_path):
    """mode='bets' CSV only includes rows where decision in ('BET', 'WATCHLIST')."""
    exporter = CsvExporter(output_dir=str(tmp_path))
    result = exporter.export(SAMPLE_BETS, mode="bets")
    path = Path(result)
    assert path.exists()

    with open(path, newline="", encoding="utf-8-sig") as f:
        # Skip comment lines
        lines = [l for l in f if not l.startswith("#")]

    reader = csv.DictReader(lines)
    rows = list(reader)
    assert len(rows) >= 1, "Expected at least one row in bets-only export"
    for row in rows:
        assert row["decision"].upper() in ("BET", "WATCHLIST"), (
            f"Unexpected decision '{row['decision']}' in bets-only export"
        )


def test_csv_summary_creates_file(tmp_path):
    """export_summary() creates a file."""
    exporter = CsvExporter(output_dir=str(tmp_path))
    result = exporter.export_summary(SAMPLE_BETS)
    assert Path(result).exists(), f"Expected summary file at {result}"


# ---------------------------------------------------------------------------
# Excel tests (skipped if openpyxl is not installed — pytest.importorskip at top)
# ---------------------------------------------------------------------------


def test_excel_export_creates_file(tmp_path):
    """ExcelExporter.export() returns a path ending in .xlsx that exists."""
    exporter = ExcelExporter(output_dir=str(tmp_path))
    result = exporter.export(SAMPLE_BETS)
    path = Path(result)
    assert path.suffix == ".xlsx", f"Expected .xlsx suffix, got '{path.suffix}'"
    assert path.exists(), f"Expected Excel file at {path} to exist"


def test_excel_export_has_bets_sheet(tmp_path):
    """The exported .xlsx file has a sheet named 'Bets'."""
    exporter = ExcelExporter(output_dir=str(tmp_path))
    result = exporter.export(SAMPLE_BETS)
    wb = openpyxl.load_workbook(result)
    assert "Bets" in wb.sheetnames, (
        f"Expected 'Bets' sheet, found: {wb.sheetnames}"
    )

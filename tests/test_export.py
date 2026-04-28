"""
Tests for export.csv_exporter (CsvExporter).
All file I/O is done to a tmp_path so no artifacts are left behind.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path


def _read_csv(path: str):
    """Read a CSV file, skipping # comment lines."""
    with open(path, encoding="utf-8-sig") as f:
        lines = [line for line in f if not line.startswith("#")]
    return list(csv.DictReader(io.StringIO("".join(lines))))


import pytest

from export.csv_exporter import CsvExporter


@pytest.fixture
def exporter(tmp_path):
    return CsvExporter(output_dir=str(tmp_path))


@pytest.fixture
def sample_records():
    return [
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
            "over_25": 0.6,
            "btts_yes": 0.5,
        },
        {
            "match_id": "1002",
            "market": "draw",
            "probability": 0.30,
            "odds": 3.8,
            "ev": 0.14,
            "kelly": 0.04,
            "kelly_stake": 10.0,
            "tier": "B",
            "model_edge": 0.02,
            "market_edge": 0.01,
            "confidence": 0.50,
            "agreement": 0.55,
            "decision": "WATCHLIST",
            "league": "Serie A",
            "over_25": 0.4,
            "btts_yes": 0.3,
        },
        {
            "match_id": "1003",
            "market": "away",
            "probability": 0.40,
            "odds": 2.8,
            "ev": -0.02,
            "kelly": 0.0,
            "kelly_stake": 0.0,
            "tier": "C",
            "model_edge": -0.01,
            "market_edge": 0.0,
            "confidence": 0.45,
            "agreement": 0.40,
            "decision": "SKIP",
            "league": "Ligue 1",
            "over_25": 0.45,
            "btts_yes": 0.35,
        },
    ]


class TestExportBetsMode:
    def test_creates_file(self, exporter, sample_records, tmp_path):
        path = exporter.export(sample_records, mode="bets")
        assert Path(path).exists()

    def test_bets_mode_filters_skip(self, exporter, sample_records, tmp_path):
        path = exporter.export(sample_records, mode="bets")
        rows = _read_csv(path)
        match_ids = [r["match_id"] for r in rows]
        assert "1001" in match_ids
        assert "1002" in match_ids
        assert "1003" not in match_ids  # SKIP decision filtered out

    def test_full_mode_includes_all(self, exporter, sample_records, tmp_path):
        path = exporter.export(sample_records, mode="full")
        rows = _read_csv(path)
        assert len(rows) == 3


class TestExportSummary:
    def test_summary_file_created(self, exporter, sample_records, tmp_path):
        path = exporter.export_summary(sample_records)
        assert Path(path).exists()

    def test_summary_has_league_rows(self, exporter, sample_records, tmp_path):
        path = exporter.export_summary(sample_records)
        rows = _read_csv(path)
        leagues = {r["league"] for r in rows}
        assert "Serie A" in leagues


class TestAppendMode:
    def test_append_creates_file(self, exporter, sample_records, tmp_path):
        path = exporter.append(sample_records[:1], "incremental.csv")
        assert Path(path).exists()

    def test_append_accumulates_rows(self, exporter, sample_records, tmp_path):
        exporter.append(sample_records[:1], "inc.csv")
        exporter.append(sample_records[1:2], "inc.csv")
        path = str(tmp_path / "inc.csv")
        with open(path, encoding="utf-8-sig") as f:
            rows = [r for r in csv.DictReader(f)]
        assert len(rows) >= 2

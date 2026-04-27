"""
Boundary value and error-path tests.

Covers edge inputs for kelly_fraction, expected_value, ProbabilityCalibration,
MatchRanker.score_one, and CsvExporter.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from ranking.match_ranker import kelly_fraction, expected_value, MatchRanker
from quant.models.calibration import ProbabilityCalibration
from export.csv_exporter import CsvExporter


# ---------------------------------------------------------------------------
# kelly_fraction edge cases
# ---------------------------------------------------------------------------


def test_kelly_fraction_extreme_probability_zero():
    """kelly_fraction(0.0, 2.0) == 0.0 — should not crash."""
    result = kelly_fraction(0.0, 2.0)
    assert result == 0.0


def test_kelly_fraction_extreme_probability_one():
    """kelly_fraction(1.0, 2.0) returns a float in [0, 1]."""
    result = kelly_fraction(1.0, 2.0)
    assert isinstance(result, float)
    assert 0.0 <= result <= 1.0


def test_kelly_fraction_minimum_odds():
    """kelly_fraction(0.5, 1.0) == 0.0 — odds=1.0 means no profit possible."""
    result = kelly_fraction(0.5, 1.0)
    assert result == 0.0


# ---------------------------------------------------------------------------
# expected_value edge cases
# ---------------------------------------------------------------------------


def test_expected_value_zero_probability():
    """expected_value(0.0, 2.0) < 0 — zero probability is always losing."""
    result = expected_value(0.0, 2.0)
    assert result < 0


# ---------------------------------------------------------------------------
# ProbabilityCalibration edge cases
# ---------------------------------------------------------------------------


def test_calibration_zero_inputs_returns_uniform():
    """calibrate_three_way({all zeros}) → all values ≈ 1/3."""
    cal = ProbabilityCalibration()
    result = cal.calibrate_three_way({"home_win": 0, "draw": 0, "away_win": 0})
    assert result["home_win"] == pytest.approx(1 / 3, abs=1e-9)
    assert result["draw"] == pytest.approx(1 / 3, abs=1e-9)
    assert result["away_win"] == pytest.approx(1 / 3, abs=1e-9)


def test_calibration_binary_clamps_below_zero():
    """calibrate_binary(-0.5) > 0 — negative inputs are clamped."""
    cal = ProbabilityCalibration()
    result = cal.calibrate_binary(-0.5)
    assert result > 0


def test_calibration_binary_clamps_above_one():
    """calibrate_binary(1.5) < 1 — inputs above 1 are clamped."""
    cal = ProbabilityCalibration()
    result = cal.calibrate_binary(1.5)
    assert result < 1


# ---------------------------------------------------------------------------
# MatchRanker.score_one edge cases
# ---------------------------------------------------------------------------


def test_match_ranker_handles_missing_fields():
    """score_one({'probability': 0.55}) returns None — no odds field."""
    ranker = MatchRanker()
    result = ranker.score_one({"probability": 0.55})
    assert result is None


def test_match_ranker_handles_nan_probability():
    """score_one with NaN probability returns None."""
    ranker = MatchRanker()
    result = ranker.score_one({"probability": float("nan"), "odds": 2.1})
    assert result is None


# ---------------------------------------------------------------------------
# CsvExporter edge cases
# ---------------------------------------------------------------------------


def test_csv_exporter_empty_records(tmp_path):
    """CsvExporter.export([], mode='full') does not raise and returns a path."""
    exporter = CsvExporter(output_dir=str(tmp_path))
    result = exporter.export([], mode="full")
    assert result is not None
    assert Path(result).exists()

"""
UI smoke tests — import-level and non-display checks.

These tests verify that UI modules are importable and that their
non-Qt logic (serialization helpers, validators, data models) works
correctly without requiring a running display or PySide6 event loop.

Tests that need a real Qt display are marked ``pytest.mark.skip``
with a note on how to run them locally:
    pytest tests/ui/ --runui    (custom marker defined in pytest.ini)
"""

from __future__ import annotations



# ---------------------------------------------------------------------------
# Import smoke — modules must be importable without a display
# ---------------------------------------------------------------------------

def test_config_settings_manager_importable():
    from config.settings_manager import load_settings, save_settings, Settings
    assert callable(load_settings)
    assert callable(save_settings)
    assert Settings is not None


def test_config_constants_importable():
    from config.constants import (
        DATABASE_NAME,
        EDGE_THRESHOLD,
        DEFAULT_LEAGUE_ID,
    )
    assert isinstance(DATABASE_NAME, str)
    assert DATABASE_NAME.endswith(".db")
    assert EDGE_THRESHOLD > 0
    assert DEFAULT_LEAGUE_ID > 0


def test_ranking_match_ranker_importable():
    from ranking.match_ranker import MatchRanker, kelly_fraction, expected_value
    assert callable(kelly_fraction)
    assert callable(expected_value)
    assert MatchRanker is not None


def test_export_modules_importable():
    from export.csv_exporter import CsvExporter
    assert CsvExporter is not None


def test_database_modules_importable():
    from database.db_manager import connect, get_db, init_db
    assert callable(connect)
    assert callable(get_db)
    assert callable(init_db)


def test_quant_models_importable():
    from quant.models.poisson_engine import PoissonEngine
    from quant.models.dixon_coles_engine import DixonColesEngine
    from quant.models.elo_engine import EloEngine
    from quant.models.form_engine import FormEngine
    from quant.models.calibration import ProbabilityCalibration
    for cls in (PoissonEngine, DixonColesEngine, EloEngine, FormEngine, ProbabilityCalibration):
        assert cls is not None


def test_analytics_modules_importable():
    from analytics.probability_markets import ProbabilityMarkets
    from analytics.league_predictability import LeaguePredictabilityAnalyser
    assert ProbabilityMarkets is not None
    assert LeaguePredictabilityAnalyser is not None


def test_engine_markowitz_importable():
    from engine.markowitz_optimizer import BetProposal, MarkowitzOptimizer, optimise_portfolio
    assert BetProposal is not None
    assert MarkowitzOptimizer is not None
    assert callable(optimise_portfolio)


# ---------------------------------------------------------------------------
# Settings data model — no display required
# ---------------------------------------------------------------------------

def test_load_settings_returns_defaults(tmp_path, monkeypatch):
    """load_settings() returns safe defaults when no file or env vars exist."""
    monkeypatch.chdir(tmp_path)
    # Clear env vars that would interfere
    monkeypatch.delenv("API_FOOTBALL_KEY", raising=False)
    monkeypatch.delenv("TELEGRAM_TOKEN", raising=False)
    from config.settings_manager import load_settings
    s = load_settings()
    assert s.league_id > 0
    assert s.season > 2000
    assert isinstance(s.api_football_key, str)


def test_save_settings_writes_only_non_secrets(tmp_path, monkeypatch):
    """save_settings() must not write api_football_key when set via env var."""
    import json
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("API_FOOTBALL_KEY", "env_key_should_not_be_written")
    from config.settings_manager import load_settings, save_settings
    s = load_settings()
    save_settings(s)
    path = tmp_path / "settings.json"
    if path.exists():
        data = json.loads(path.read_text())
        assert "api_football_key" not in data, "env-backed secret must not be written"


def test_save_settings_never_writes_credentials(tmp_path, monkeypatch):
    """save_settings() must NEVER write credentials to disk — security policy."""
    import json
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("API_FOOTBALL_KEY", raising=False)
    # Pre-seed the file with a credential (simulating a legacy settings.json)
    (tmp_path / "settings.json").write_text(
        json.dumps({"league_id": 135, "season": 2024, "api_football_key": "file_key"})
    )
    from config.settings_manager import load_settings, save_settings
    s = load_settings()
    save_settings(s)
    data = json.loads((tmp_path / "settings.json").read_text())
    # Credentials must NOT be written back regardless of whether env var is set
    assert "api_football_key" not in data, (
        "save_settings() must never persist credentials to disk"
    )


# ---------------------------------------------------------------------------
# Constants — DB path is user-writable
# ---------------------------------------------------------------------------

def test_database_name_not_in_source_tree():
    """DATABASE_NAME must not resolve inside the repo (read-only install safety)."""
    import os
    from config.constants import DATABASE_NAME
    # Either env override or user home dir — either way not in CWD/source
    env_override = os.getenv("FQE_DB_PATH")
    if not env_override:
        assert ".footballquantengine" in DATABASE_NAME or "FQE_DB_PATH" in os.environ

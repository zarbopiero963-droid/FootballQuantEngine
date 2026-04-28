import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pytest
import sqlite3
from unittest.mock import MagicMock

SCHEMA_PATH = ROOT_DIR / "database" / "schema.sql"


# ---------------------------------------------------------------------------
# Database fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def in_memory_db():
    """In-memory SQLite connection with the full schema applied."""
    conn = sqlite3.connect(":memory:")
    with open(SCHEMA_PATH, encoding="utf-8") as f:
        conn.executescript(f.read())
    yield conn
    conn.close()


# ---------------------------------------------------------------------------
# Match data fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_matches():
    """Five completed match dicts (status FT)."""
    return [
        {
            "fixture_id": "1001",
            "home_team": "Juventus",
            "away_team": "Inter",
            "home_goals": 2,
            "away_goals": 1,
            "match_date": "2024-01-15T15:00:00+00:00",
            "status": "FT",
            "league_id": 135,
            "league": "Serie A",
            "season": 2024,
            "home_corners": 6,
            "away_corners": 4,
            "home_shots": 14,
            "away_shots": 8,
            "home_xg": 1.8,
            "away_xg": 0.9,
            "referee": "Orsato",
        },
        {
            "fixture_id": "1002",
            "home_team": "AC Milan",
            "away_team": "Roma",
            "home_goals": 0,
            "away_goals": 0,
            "match_date": "2024-01-16T18:00:00+00:00",
            "status": "FT",
            "league_id": 135,
            "league": "Serie A",
            "season": 2024,
            "home_corners": 5,
            "away_corners": 5,
            "home_shots": 10,
            "away_shots": 11,
            "home_xg": 1.1,
            "away_xg": 1.0,
            "referee": "Mariani",
        },
        {
            "fixture_id": "1003",
            "home_team": "Napoli",
            "away_team": "Lazio",
            "home_goals": 3,
            "away_goals": 1,
            "match_date": "2024-01-20T15:00:00+00:00",
            "status": "FT",
            "league_id": 135,
            "league": "Serie A",
            "season": 2024,
            "home_corners": 8,
            "away_corners": 3,
            "home_shots": 17,
            "away_shots": 7,
            "home_xg": 2.4,
            "away_xg": 0.7,
            "referee": "Di Bello",
        },
        {
            "fixture_id": "1004",
            "home_team": "Atalanta",
            "away_team": "Fiorentina",
            "home_goals": 1,
            "away_goals": 2,
            "match_date": "2024-01-21T20:45:00+00:00",
            "status": "FT",
            "league_id": 135,
            "league": "Serie A",
            "season": 2024,
            "home_corners": 4,
            "away_corners": 7,
            "home_shots": 9,
            "away_shots": 13,
            "home_xg": 0.9,
            "away_xg": 1.6,
            "referee": "Sozza",
        },
        {
            "fixture_id": "1005",
            "home_team": "Bologna",
            "away_team": "Torino",
            "home_goals": 2,
            "away_goals": 2,
            "match_date": "2024-01-22T18:00:00+00:00",
            "status": "FT",
            "league_id": 135,
            "league": "Serie A",
            "season": 2024,
            "home_corners": 6,
            "away_corners": 6,
            "home_shots": 12,
            "away_shots": 12,
            "home_xg": 1.5,
            "away_xg": 1.5,
            "referee": "Rocchi",
        },
    ]


@pytest.fixture
def sample_upcoming():
    """Three upcoming match dicts (status NS, no goals/corners)."""
    return [
        {
            "fixture_id": "2001",
            "home_team": "Juventus",
            "away_team": "Napoli",
            "match_date": "2024-02-10T15:00:00+00:00",
            "status": "NS",
            "league_id": 135,
            "league": "Serie A",
            "season": 2024,
        },
        {
            "fixture_id": "2002",
            "home_team": "Inter",
            "away_team": "AC Milan",
            "match_date": "2024-02-11T18:00:00+00:00",
            "status": "NS",
            "league_id": 135,
            "league": "Serie A",
            "season": 2024,
        },
        {
            "fixture_id": "2003",
            "home_team": "Roma",
            "away_team": "Lazio",
            "match_date": "2024-02-12T20:45:00+00:00",
            "status": "NS",
            "league_id": 135,
            "league": "Serie A",
            "season": 2024,
        },
    ]


# ---------------------------------------------------------------------------
# Mock API client fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_api_client(sample_matches, sample_upcoming):
    """MagicMock pre-configured with realistic return values."""
    client = MagicMock()
    client.get_completed_matches.return_value = sample_matches
    client.get_upcoming_matches.return_value = sample_upcoming
    client.get_prematch_odds.return_value = {
        "2001": {"1x2": {"home": 2.1, "draw": 3.4, "away": 3.2}},
        "2002": {"1x2": {"home": 2.1, "draw": 3.4, "away": 3.2}},
        "2003": {"1x2": {"home": 2.1, "draw": 3.4, "away": 3.2}},
    }
    client.get_available_seasons.return_value = [2022, 2023, 2024]
    client.get_fixture_stats.return_value = {
        "fixture_id": "1001",
        "home_corners": 6,
        "away_corners": 4,
    }
    return client


# ---------------------------------------------------------------------------
# Ranked bets fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_ranked_bets():
    """Four dicts matching RankedBet.as_dict() format."""
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
        },
        {
            "match_id": "1002",
            "market": "away",
            "probability": 0.48,
            "odds": 2.3,
            "ev": 0.104,
            "kelly": 0.055,
            "kelly_stake": 13.75,
            "tier": "B",
            "model_edge": 0.03,
            "market_edge": 0.01,
            "confidence": 0.58,
            "agreement": 0.60,
            "decision": "BET",
            "league": "Serie A",
        },
        {
            "match_id": "1003",
            "market": "home",
            "probability": 0.62,
            "odds": 1.9,
            "ev": 0.118,
            "kelly": 0.097,
            "kelly_stake": 24.25,
            "tier": "A",
            "model_edge": 0.07,
            "market_edge": 0.04,
            "confidence": 0.70,
            "agreement": 0.80,
            "decision": "BET",
            "league": "Serie A",
        },
        {
            "match_id": "1004",
            "market": "draw",
            "probability": 0.35,
            "odds": 3.1,
            "ev": 0.085,
            "kelly": 0.040,
            "kelly_stake": 10.0,
            "tier": "B",
            "model_edge": 0.02,
            "market_edge": 0.00,
            "confidence": 0.50,
            "agreement": 0.55,
            "decision": "BET",
            "league": "Serie A",
        },
    ]

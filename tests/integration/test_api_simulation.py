"""
Integration tests simulating the full api-football.com pipeline using
realistic mock data that mirrors the real v3 API JSON structure.

Run with: pytest tests/integration/ -v -m integration
Skip in fast CI: pytest -m "not integration"
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

pytestmark = pytest.mark.integration

# ---------------------------------------------------------------------------
# Realistic API-Football v3 response payloads
# ---------------------------------------------------------------------------

FIXTURE_RESPONSE = {
    "response": [
        {
            "fixture": {
                "id": 1001,
                "date": "2024-03-15T15:00:00+00:00",
                "status": {"short": "FT", "elapsed": 90},
                "venue": {"name": "Allianz Stadium"},
                "referee": "Daniele Orsato",
            },
            "league": {"id": 135, "name": "Serie A", "season": 2024},
            "teams": {
                "home": {"id": 496, "name": "Juventus"},
                "away": {"id": 489, "name": "Inter"},
            },
            "goals": {"home": 2, "away": 1},
            "score": {"halftime": {"home": 1, "away": 0}},
        }
    ]
}

UPCOMING_RESPONSE = {
    "response": [
        {
            "fixture": {
                "id": 2001,
                "date": "2024-04-01T20:45:00+00:00",
                "status": {"short": "NS", "elapsed": None},
                "venue": {"name": "Allianz Stadium"},
                "referee": None,
            },
            "league": {"id": 135, "name": "Serie A", "season": 2024},
            "teams": {
                "home": {"id": 496, "name": "Juventus"},
                "away": {"id": 492, "name": "Napoli"},
            },
            "goals": {"home": None, "away": None},
            "score": {"halftime": {"home": None, "away": None}},
        }
    ]
}

ODDS_RESPONSE = {
    "response": [
        {
            "fixture": {"id": 2001},
            "bookmakers": [
                {
                    "id": 6,
                    "name": "Bwin",
                    "bets": [
                        {
                            "id": 1,
                            "name": "Match Winner",
                            "values": [
                                {"value": "Home", "odd": "2.10"},
                                {"value": "Draw", "odd": "3.40"},
                                {"value": "Away", "odd": "3.20"},
                            ],
                        },
                        {
                            "id": 5,
                            "name": "Goals Over/Under",
                            "values": [
                                {"value": "Over 2.5", "odd": "1.85"},
                                {"value": "Under 2.5", "odd": "1.95"},
                            ],
                        },
                        {
                            "id": 8,
                            "name": "Both Teams Score",
                            "values": [
                                {"value": "Yes", "odd": "1.75"},
                                {"value": "No", "odd": "1.95"},
                            ],
                        },
                    ],
                }
            ],
        }
    ]
}

STANDINGS_RESPONSE = {
    "response": [
        {
            "league": {
                "standings": [
                    [
                        {
                            "rank": 1,
                            "team": {"id": 496, "name": "Juventus"},
                            "points": 60,
                            "goalsDiff": 25,
                            "all": {"win": 19, "lose": 3, "draw": 4},
                        },
                        {
                            "rank": 2,
                            "team": {"id": 489, "name": "Inter"},
                            "points": 58,
                            "goalsDiff": 22,
                            "all": {"win": 18, "lose": 4, "draw": 4},
                        },
                    ]
                ]
            }
        }
    ]
}


@pytest.fixture
def mock_resp():
    def _make(payload):
        r = MagicMock()
        r.raise_for_status.return_value = None
        r.json.return_value = payload
        r.status_code = 200
        return r

    return _make


def test_parse_fixture_from_api_response(mock_resp):
    with patch("requests.get", return_value=mock_resp(FIXTURE_RESPONSE)):
        from quant.providers.api_football_client import APIFootballClient

        client = APIFootballClient(api_key="test_key")
        matches = client.get_completed_matches(135, 2024)

    assert len(matches) == 1
    m = matches[0]
    assert m["home_team"] == "Juventus"
    assert m["away_team"] == "Inter"
    assert m["home_goals"] == 2
    assert m["away_goals"] == 1
    assert m["fixture_id"] == "1001"
    assert m["referee"] == "Daniele Orsato"


def test_parse_upcoming_fixture(mock_resp):
    with patch("requests.get", return_value=mock_resp(UPCOMING_RESPONSE)):
        from quant.providers.api_football_client import APIFootballClient

        client = APIFootballClient(api_key="test_key")
        upcoming = client.get_upcoming_matches(135, 2024)

    assert len(upcoming) == 1
    assert upcoming[0]["home_team"] == "Juventus"
    assert upcoming[0]["away_team"] == "Napoli"


def test_parse_prematch_odds_1x2(mock_resp):
    with patch("requests.get", return_value=mock_resp(ODDS_RESPONSE)):
        from quant.providers.api_football_client import APIFootballClient

        client = APIFootballClient(api_key="test_key")
        odds = client.get_prematch_odds([2001])

    x12 = odds["2001"]["1x2"]
    assert x12["home"] == pytest.approx(2.10)
    assert x12["draw"] == pytest.approx(3.40)
    assert x12["away"] == pytest.approx(3.20)


def test_parse_prematch_odds_all_markets_present(mock_resp):
    """Client parses 1x2, ou25, and btts markets from the same response."""
    with patch("requests.get", return_value=mock_resp(ODDS_RESPONSE)):
        from quant.providers.api_football_client import APIFootballClient

        client = APIFootballClient(api_key="test_key")
        odds = client.get_prematch_odds([2001])

    markets = odds["2001"]
    assert "1x2" in markets
    assert "ou25" in markets
    assert "btts" in markets


def test_parse_prematch_odds_missing_fixture(mock_resp):
    """Unknown fixture_id returns empty dict (no KeyError)."""
    with patch("requests.get", return_value=mock_resp({"response": []})):
        from quant.providers.api_football_client import APIFootballClient

        client = APIFootballClient(api_key="test_key")
        odds = client.get_prematch_odds([9999])

    assert odds == {}


def test_parse_standings(mock_resp):
    with patch("requests.get", return_value=mock_resp(STANDINGS_RESPONSE)):
        from quant.providers.api_football_client import APIFootballClient

        client = APIFootballClient(api_key="test_key")
        standings = client.get_standings(135, 2024)

    assert standings[0]["team"] == "Juventus"
    assert standings[0]["position"] == 1
    assert standings[0]["points"] == 60
    assert standings[1]["team"] == "Inter"
    assert standings[1]["points"] == 58


def test_full_pipeline_rank_and_export(tmp_path):
    """Fixture data → MatchRanker → CsvExporter pipeline."""
    from ranking.match_ranker import MatchRanker
    from export.csv_exporter import CsvExporter

    records = [
        {
            "match_id": "2001",
            "league_id": 135,
            "league_name": "Serie A",
            "league": "Serie A",
            "home_team": "Juventus",
            "away_team": "Napoli",
            "market": "home",
            "probability": 0.55,
            "fair_odds": 1.82,
            "odds": 2.10,
            "market_edge": 0.038,
            "model_edge": 0.055,
            "confidence": 0.67,
            "agreement": 0.72,
            "decision": "BET",
        }
    ]

    ranked = MatchRanker(bankroll=1000.0, kelly_scale=0.25).rank_as_dicts(records)
    assert len(ranked) >= 1
    assert ranked[0]["tier"] in ("S", "A", "B", "C")

    csv_path = CsvExporter(output_dir=str(tmp_path)).export(ranked, mode="full")
    assert Path(csv_path).exists()
    assert Path(csv_path).stat().st_size > 0

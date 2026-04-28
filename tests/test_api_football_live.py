from __future__ import annotations

from unittest.mock import MagicMock, patch

from quant.providers.api_football_client import APIFootballClient


def _mock_response(json_data: dict) -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = json_data
    resp.raise_for_status = MagicMock()
    return resp


@patch("quant.providers.api_football_client.requests.get")
def test_api_connection(mock_get):
    mock_get.return_value = _mock_response(
        {
            "response": [
                {"league": {"id": 135, "name": "Serie A", "country": "Italy"}},
                {"league": {"id": 39, "name": "Premier League", "country": "England"}},
            ]
        }
    )

    client = APIFootballClient(api_key="test-key-ci")
    leagues = client.get_leagues()

    assert isinstance(leagues, list)
    assert len(leagues) >= 1
    mock_get.assert_called_once()

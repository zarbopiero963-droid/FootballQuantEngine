from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests

from quant.providers.api_football_client import APIFootballClient

_RESPONSE_JSON = {
    "response": [
        {"league": {"id": 135, "name": "Serie A", "country": "Italy"}},
        {"league": {"id": 39, "name": "Premier League", "country": "England"}},
    ]
}


def _mock_response(
    json_data: dict,
    status_code: int = 200,
    raise_for_status_side_effect=None,
) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.raise_for_status = MagicMock(side_effect=raise_for_status_side_effect)
    return resp


@patch("quant.providers.api_football_client.requests.get")
def test_api_connection(mock_get):
    mock_get.return_value = _mock_response(_RESPONSE_JSON)

    client = APIFootballClient(api_key="test-key-ci")
    leagues = client.get_leagues()

    assert isinstance(leagues, list)
    assert len(leagues) == 2
    assert leagues == _RESPONSE_JSON["response"]
    assert leagues[0]["league"]["id"] == 135
    assert leagues[0]["league"]["name"] == "Serie A"
    assert leagues[0]["league"]["country"] == "Italy"
    assert leagues[1]["league"]["id"] == 39
    assert leagues[1]["league"]["name"] == "Premier League"
    assert leagues[1]["league"]["country"] == "England"

    mock_get.assert_called_once()
    args, kwargs = mock_get.call_args
    assert "leagues" in args[0]
    headers = kwargs.get("headers", {})
    assert any(k.lower() == "x-apisports-key" for k in headers)
    assert headers[[k for k in headers if k.lower() == "x-apisports-key"][0]] == "test-key-ci"
    params = kwargs.get("params")
    if params is not None:
        assert isinstance(params, dict)


@patch("quant.providers.api_football_client.requests.get")
def test_api_connection_error(mock_get):
    mock_get.return_value = _mock_response(
        json_data={},
        status_code=500,
        raise_for_status_side_effect=requests.HTTPError("Server error"),
    )

    client = APIFootballClient(api_key="test-key-ci")

    with pytest.raises(requests.HTTPError):
        client.get_leagues()

    mock_get.assert_called_once()

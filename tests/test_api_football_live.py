import os

import pytest

if not os.getenv("API_FOOTBALL_KEY"):
    pytest.skip("API_FOOTBALL_KEY missing", allow_module_level=True)

from quant.providers.api_football_client import APIFootballClient


def test_api_connection():

    client = APIFootballClient()

    leagues = client.get_leagues()

    assert isinstance(leagues, list)

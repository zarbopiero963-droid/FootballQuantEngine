"""
Tests for database.fixtures_repository — uses the in_memory_db fixture from conftest.py.
"""

from __future__ import annotations

import sqlite3
from unittest.mock import patch, MagicMock

import pytest

from database.fixtures_repository import FixturesRepository


class _NoCloseProxy:
    """Forward all attribute access/assignment to the real connection; stub close()."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        object.__setattr__(self, "_conn", conn)

    def __getattr__(self, name: str):
        return getattr(object.__getattribute__(self, "_conn"), name)

    def __setattr__(self, name: str, value) -> None:
        setattr(object.__getattribute__(self, "_conn"), name, value)

    def close(self) -> None:
        pass  # no-op — keep the in-memory DB alive for the entire test


@pytest.fixture
def repo(in_memory_db):
    """FixturesRepository connected to the in-memory DB via _NoCloseProxy."""
    proxy = _NoCloseProxy(in_memory_db)
    with patch("database.fixtures_repository.connect", return_value=proxy):
        yield FixturesRepository()


_FIXTURE = {
    "fixture_id": "9001",
    "home_team": "Juventus",
    "away_team": "Inter",
    "home_goals": 2,
    "away_goals": 1,
    "match_date": "2024-03-01T15:00:00+00:00",
    "status": "FT",
    "league_id": 135,
    "league": "Serie A",
    "season": 2024,
}


class TestUpsertAndRetrieve:
    def _fids(self, rows):
        """Normalize fixture_id to str for comparison."""
        return [str(r["fixture_id"]) for r in rows]

    def test_upsert_stores_fixture(self, repo):
        repo.upsert_fixture(_FIXTURE)
        rows = repo.get_fixtures_by_status("FT")
        assert "9001" in self._fids(rows)

    def test_upsert_is_idempotent(self, repo):
        repo.upsert_fixture(_FIXTURE)
        repo.upsert_fixture(_FIXTURE)
        rows = repo.get_fixtures_by_status("FT")
        assert self._fids(rows).count("9001") == 1

    def test_get_by_status_filters_correctly(self, repo):
        repo.upsert_fixture(_FIXTURE)
        repo.upsert_fixture({**_FIXTURE, "fixture_id": "9002", "status": "NS"})
        ft = repo.get_fixtures_by_status("FT")
        ns = repo.get_fixtures_by_status("NS")
        assert all(r["status"] == "FT" for r in ft)
        assert all(r["status"] == "NS" for r in ns)


class TestBulkUpsert:
    def test_bulk_inserts_all(self, repo):
        fixtures = [
            {**_FIXTURE, "fixture_id": f"80{i:02d}", "home_goals": i}
            for i in range(5)
        ]
        count = repo.upsert_fixtures_bulk(fixtures)
        assert count == 5

    def test_bulk_empty_list(self, repo):
        count = repo.upsert_fixtures_bulk([])
        assert count == 0


class TestSeasonRegistry:
    def test_season_not_downloaded_initially(self, repo):
        assert not repo.is_season_downloaded(135, 2025)

    def test_mark_and_check_downloaded(self, repo):
        repo.mark_season_downloaded(135, 2024, 380)
        assert repo.is_season_downloaded(135, 2024)

    def test_downloaded_seasons_for_league(self, repo):
        repo.mark_season_downloaded(135, 2023, 380)
        repo.mark_season_downloaded(135, 2024, 380)
        seasons = repo.downloaded_seasons_for_league(135)
        assert 2023 in seasons
        assert 2024 in seasons

"""
Unit tests for database/db_manager.py and database/schema.sql.

All tests use an in-memory SQLite database; the real DB file on disk is
never touched.  No API keys are required.
"""

import sqlite3
import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
SCHEMA_PATH = ROOT_DIR / "database" / "schema.sql"

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_schema(conn: sqlite3.Connection) -> None:
    """Execute the project schema script against *conn*."""
    with open(SCHEMA_PATH, encoding="utf-8") as f:
        conn.executescript(f.read())


def _get_table_names(conn: sqlite3.Connection) -> set:
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    return {row[0] for row in cur.fetchall()}


# ---------------------------------------------------------------------------
# Test 1 – schema creates the expected tables
# ---------------------------------------------------------------------------


def test_init_db_creates_all_tables():
    """
    Loading the schema into an in-memory DB must create every table defined
    in schema.sql.

    Rather than calling init_db() (which hard-codes a file path), we open a
    fresh ':memory:' connection and run the schema script directly — this is
    the most reliable approach for in-memory testing.
    """
    conn = sqlite3.connect(":memory:")
    try:
        _load_schema(conn)
        tables = _get_table_names(conn)
    finally:
        conn.close()

    expected = {"fixtures", "snapshots", "downloaded_seasons", "predictions", "odds_history"}
    assert expected.issubset(tables), (
        f"Missing tables: {expected - tables}.  Found: {tables}"
    )


# ---------------------------------------------------------------------------
# Test 2 – get_db() commits on normal exit
# ---------------------------------------------------------------------------


def test_get_db_commits_on_success(monkeypatch):
    """
    A row inserted inside a 'with get_db()' block must be visible after the
    block exits normally (i.e. the connection was committed).
    """
    import database.db_manager as dm

    # Redirect every connect() call to a shared in-memory connection so that
    # we can inspect the committed data after get_db() closes its connection.
    # SQLite ':memory:' databases are normally per-connection, so we use a
    # named shared-cache URI to allow multiple handles to the same DB.
    shared_uri = "file:test_commit?mode=memory&cache=shared"

    # Set up the shared DB with a test table.
    setup_conn = sqlite3.connect(shared_uri, uri=True)
    setup_conn.execute("CREATE TABLE IF NOT EXISTS _test (val TEXT)")
    setup_conn.commit()

    monkeypatch.setattr(dm, "DATABASE_NAME", shared_uri)
    # connect() must use uri=True for shared-cache URIs.
    def _uri_connect():
        return sqlite3.connect(dm.DATABASE_NAME, uri=True)

    monkeypatch.setattr(dm, "connect", _uri_connect)

    try:
        with dm.get_db() as conn:
            conn.execute("INSERT INTO _test (val) VALUES ('committed')")

        # Verify via the setup connection.
        row = setup_conn.execute("SELECT val FROM _test").fetchone()
        assert row is not None, "Row should have been committed but was not found"
        assert row[0] == "committed"
    finally:
        setup_conn.close()


# ---------------------------------------------------------------------------
# Test 3 – get_db() rolls back on exception
# ---------------------------------------------------------------------------


def test_get_db_rolls_back_on_exception(monkeypatch):
    """
    A row inserted inside a 'with get_db()' block must NOT be visible if an
    exception is raised inside the block (i.e. the connection was rolled back).
    """
    import database.db_manager as dm

    shared_uri = "file:test_rollback?mode=memory&cache=shared"

    setup_conn = sqlite3.connect(shared_uri, uri=True)
    setup_conn.execute("CREATE TABLE IF NOT EXISTS _test (val TEXT)")
    setup_conn.commit()

    monkeypatch.setattr(dm, "DATABASE_NAME", shared_uri)

    def _uri_connect():
        return sqlite3.connect(dm.DATABASE_NAME, uri=True)

    monkeypatch.setattr(dm, "connect", _uri_connect)

    try:
        with pytest.raises(RuntimeError):
            with dm.get_db() as conn:
                conn.execute("INSERT INTO _test (val) VALUES ('should_rollback')")
                raise RuntimeError("deliberate error to trigger rollback")

        row = setup_conn.execute("SELECT val FROM _test WHERE val='should_rollback'").fetchone()
        assert row is None, "Row should have been rolled back but was found in the DB"
    finally:
        setup_conn.close()


# ---------------------------------------------------------------------------
# Test 4 – connect() returns a sqlite3.Connection
# ---------------------------------------------------------------------------


def test_connect_returns_connection(monkeypatch):
    """
    connect() must return a sqlite3.Connection instance.
    Monkeypatched to ':memory:' so no real DB file is created.
    """
    import database.db_manager as dm

    monkeypatch.setattr(dm, "DATABASE_NAME", ":memory:")

    conn = dm.connect()
    try:
        assert isinstance(conn, sqlite3.Connection), (
            f"Expected sqlite3.Connection, got {type(conn)}"
        )
    finally:
        conn.close()

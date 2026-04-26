import sqlite3
from pathlib import Path

from config.constants import DATABASE_NAME

BASE_DIR = Path(__file__).resolve().parent
SCHEMA_PATH = BASE_DIR / "schema.sql"

# Columns added to fixtures in schema v2; ALTER TABLE is a no-op if already present.
_FIXTURES_V2_COLUMNS = [
    ("league_id",  "INTEGER"),
    ("ht_home",    "INTEGER"),
    ("ht_away",    "INTEGER"),
    ("venue",      "TEXT"),
    ("referee",    "TEXT"),
    ("elapsed",    "INTEGER"),
    ("updated_at", "TEXT"),
]

# Column added to odds_history in schema v2.
_ODDS_HISTORY_V2_COLUMNS = [
    ("market", "TEXT DEFAULT '1x2'"),
]


def connect():
    return sqlite3.connect(DATABASE_NAME)


def init_db():
    conn = connect()

    with open(SCHEMA_PATH, encoding="utf-8") as f:
        conn.executescript(f.read())

    conn.commit()
    _migrate(conn)
    conn.close()


def _migrate(conn: sqlite3.Connection) -> None:
    """Add columns introduced in v2 to pre-existing databases.

    SQLite does not support IF NOT EXISTS on ALTER TABLE, so we catch the
    OperationalError that is raised when a column already exists.
    """
    for col, col_type in _FIXTURES_V2_COLUMNS:
        try:
            conn.execute(f"ALTER TABLE fixtures ADD COLUMN {col} {col_type}")
        except sqlite3.OperationalError:
            pass  # column already exists

    for col, col_type in _ODDS_HISTORY_V2_COLUMNS:
        try:
            conn.execute(f"ALTER TABLE odds_history ADD COLUMN {col} {col_type}")
        except sqlite3.OperationalError:
            pass

    conn.commit()

import logging
import sqlite3
from pathlib import Path

from config.constants import DATABASE_NAME

BASE_DIR = Path(__file__).resolve().parent
SCHEMA_PATH = BASE_DIR / "schema.sql"

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Migration registry
# Each entry: (version, human-readable description, list of (table, col, type))
# Only ADD to this list; never edit or remove an existing entry.
# ---------------------------------------------------------------------------
_MIGRATIONS: list[tuple[int, str, list[tuple[str, str, str]]]] = [
    (
        2,
        "Extended fixtures with league_id, halftime scores, venue, referee; "
        "added market column to odds_history",
        [
            ("fixtures",     "league_id",  "INTEGER"),
            ("fixtures",     "ht_home",    "INTEGER"),
            ("fixtures",     "ht_away",    "INTEGER"),
            ("fixtures",     "venue",      "TEXT"),
            ("fixtures",     "referee",    "TEXT"),
            ("fixtures",     "elapsed",    "INTEGER"),
            ("fixtures",     "updated_at", "TEXT"),
            ("odds_history", "market",     "TEXT DEFAULT '1x2'"),
        ],
    ),
    (
        3,
        "Added league_name column to downloaded_seasons for human-readable display",
        [
            ("downloaded_seasons", "league_name", "TEXT NOT NULL DEFAULT ''"),
        ],
    ),
]


def connect() -> sqlite3.Connection:
    return sqlite3.connect(DATABASE_NAME)


def init_db() -> None:
    conn = connect()

    with open(SCHEMA_PATH, encoding="utf-8") as f:
        conn.executescript(f.read())

    conn.commit()
    _run_migrations(conn)
    conn.close()


def _run_migrations(conn: sqlite3.Connection) -> None:
    """Apply pending schema migrations in version order.

    Uses the schema_migrations table (created by schema.sql) to track which
    versions have already been applied, so each migration runs exactly once.
    SQLite does not support ALTER TABLE … ADD COLUMN IF NOT EXISTS, so we
    catch the OperationalError that fires when a column already exists.
    Any other OperationalError (missing table, lock, etc.) is re-raised so
    the migration is NOT recorded as applied and will be retried next time.
    """
    applied: set[int] = {
        row[0]
        for row in conn.execute("SELECT version FROM schema_migrations").fetchall()
    }

    for version, description, columns in _MIGRATIONS:
        if version in applied:
            continue

        logger.info("Applying schema migration v%d: %s", version, description)
        for table, col, col_type in columns:
            try:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {col_type}")
            except sqlite3.OperationalError as exc:
                if "duplicate column name" not in str(exc).lower():
                    raise  # unexpected error — do not mark migration as applied

        conn.execute(
            "INSERT INTO schema_migrations(version, description) VALUES(?, ?)",
            (version, description),
        )
        conn.commit()
        logger.info("Migration v%d applied.", version)

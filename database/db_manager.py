import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from config.constants import DATABASE_NAME

BASE_DIR = Path(__file__).resolve().parent
SCHEMA_PATH = BASE_DIR / "schema.sql"

logger = logging.getLogger(__name__)


def connect() -> sqlite3.Connection:
    return sqlite3.connect(DATABASE_NAME)


@contextmanager
def get_db():
    """
    Context manager that yields a committed-or-rolled-back connection.

    Usage::

        with get_db() as conn:
            conn.execute("INSERT INTO ...")
        # auto-committed on success, rolled back on exception
    """
    conn = connect()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    conn = connect()

    with open(SCHEMA_PATH, encoding="utf-8") as f:
        conn.executescript(f.read())

    conn.commit()
    conn.close()

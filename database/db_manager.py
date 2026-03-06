import sqlite3
from pathlib import Path

from config.constants import DATABASE_NAME

BASE_DIR = Path(__file__).resolve().parent
SCHEMA_PATH = BASE_DIR / "schema.sql"


def connect():

    return sqlite3.connect(DATABASE_NAME)


def init_db():

    conn = connect()

    with open(SCHEMA_PATH, encoding="utf-8") as f:
        conn.executescript(f.read())

    conn.commit()
    conn.close()

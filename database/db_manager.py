import sqlite3

from config.constants import DATABASE_NAME


def connect():

    return sqlite3.connect(DATABASE_NAME)


def init_db():

    conn = connect()

    with open("database/schema.sql") as f:
        conn.executescript(f.read())

    conn.commit()
    conn.close()

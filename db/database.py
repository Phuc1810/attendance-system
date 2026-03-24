import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "attendance.db"


def connect_db():
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def initialize_database():
    with connect_db() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS employees (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                department TEXT NOT NULL
            )
            """
        )

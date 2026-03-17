import sqlite3

def connect_db():
    conn = sqlite3.connect("attendance.db")
    return conn

def create_table():
    conn = connect_db()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS employees (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        department TEXT
    )
    """)

    conn.commit()
    conn.close()

create_table()

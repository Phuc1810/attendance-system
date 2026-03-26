import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st

#Thiết lập kết nối và kết nối database
DB_PATH = Path(__file__).resolve().parent.parent / "attendance.db"
EMPLOYEE_CODE_PREFIX = "NV"
EMPLOYEE_CODE_WIDTH = 3


def connect_db():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

#Thuật toán xử lý mã nhân viên
def format_employee_code(number):
    return f"{EMPLOYEE_CODE_PREFIX}{number:0{EMPLOYEE_CODE_WIDTH}d}"


def parse_employee_code(employee_code):
    if not employee_code or not employee_code.startswith(EMPLOYEE_CODE_PREFIX):
        return None

    number_part = employee_code[len(EMPLOYEE_CODE_PREFIX):]
    if not number_part.isdigit():
        return None

    return int(number_part)


def _column_exists(conn, table_name, column_name):
    columns = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    return any(column[1] == column_name for column in columns)


def _get_next_employee_code(conn):
    rows = conn.execute("SELECT employee_code FROM employees WHERE employee_code IS NOT NULL").fetchall()
    used_numbers = {
        number
        for (employee_code,) in rows
        for number in [parse_employee_code(employee_code)]
        if number is not None
    }

    next_number = 1
    while next_number in used_numbers:
        next_number += 1

    return format_employee_code(next_number)


def _assign_missing_employee_codes(conn):
    rows = conn.execute("SELECT id, employee_code FROM employees ORDER BY id").fetchall()
    used_numbers = {
        number
        for _, employee_code in rows
        for number in [parse_employee_code(employee_code)]
        if number is not None
    }

    next_number = 1
    for employee_id, employee_code in rows:
        if employee_code:
            continue

        while next_number in used_numbers:
            next_number += 1

        new_code = format_employee_code(next_number)
        conn.execute(
            "UPDATE employees SET employee_code = ? WHERE id = ?",
            (new_code, employee_id),
        )
        used_numbers.add(next_number)
        next_number += 1


def initialize_database():
    with connect_db() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS employees (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                employee_code TEXT UNIQUE,
                name TEXT NOT NULL,
                department TEXT NOT NULL
            )
            """
        )

        if not _column_exists(conn, "employees", "employee_code"):
            conn.execute("ALTER TABLE employees ADD COLUMN employee_code TEXT")

        _assign_missing_employee_codes(conn)
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_employees_employee_code ON employees(employee_code)"
        )
        employee_rows = conn.execute(
            "SELECT id, employee_code FROM employees ORDER BY id"
        ).fetchall()

    from core.save_face import sync_employee_face_folders

    sync_employee_face_folders(employee_rows)


def get_employee_codes():
    initialize_database()

    with connect_db() as conn:
        rows = conn.execute(
            "SELECT employee_code FROM employees WHERE employee_code IS NOT NULL ORDER BY id"
        ).fetchall()

    return [employee_code for (employee_code,) in rows]


@st.cache_data(show_spinner=False)
def get_all_employees():
    with connect_db() as conn:
        return conn.execute(
            "SELECT id, employee_code, name, department FROM employees ORDER BY id"
        ).fetchall()


@st.cache_data(show_spinner=False)
def get_employee_dataframe():
    with connect_db() as conn:
        return pd.read_sql(
            "SELECT employee_code AS code, name, department, id FROM employees ORDER BY id",
            conn,
        )


def clear_employee_cache():
    get_all_employees.clear()
    get_employee_dataframe.clear()


def create_employee(name, department):
    initialize_database()

    with connect_db() as conn:
        employee_code = _get_next_employee_code(conn)
        conn.execute(
            "INSERT INTO employees (employee_code, name, department) VALUES (?, ?, ?)",
            (employee_code, name.strip(), department.strip()),
        )

    from core.save_face import ensure_employee_folder

    ensure_employee_folder(employee_code)
    clear_employee_cache()
    return employee_code

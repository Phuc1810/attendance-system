from datetime import datetime

import pandas as pd
import streamlit as st

from db.database import connect_db

VALID_LOG_TYPES = {"IN", "OUT"}


def initialize_attendance_logs():
    with connect_db() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS attendance_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                employee_code TEXT NOT NULL,
                log_time TEXT NOT NULL,
                log_type TEXT NOT NULL,
                camera_source TEXT,
                confidence REAL
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_attendance_logs_employee_time ON attendance_logs(employee_code, log_time DESC)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_attendance_logs_time ON attendance_logs(log_time DESC)"
        )


def _normalize_log_type(log_type):
    normalized_log_type = str(log_type).upper().strip()

    if normalized_log_type not in VALID_LOG_TYPES:
        raise ValueError("log_type must be 'IN' or 'OUT'.")

    return normalized_log_type


def _normalize_timestamp(reference_time=None):
    timestamp = reference_time or datetime.now()
    return timestamp.strftime("%Y-%m-%d %H:%M:%S")


def _normalize_date(reference_time=None):
    timestamp = reference_time or datetime.now()
    return timestamp.strftime("%Y-%m-%d")


def _row_to_dict(row):
    if row is None:
        return None

    return {
        "id": row[0],
        "employee_code": row[1],
        "log_time": row[2],
        "log_type": row[3],
        "camera_source": row[4],
        "confidence": row[5],
    }


def clear_attendance_cache():
    get_latest_attendance_log.clear()
    get_attendance_count.clear()
    get_today_attendance_dataframe.clear()
    get_attendance_history_dataframe.clear()


def create_attendance_log(
    employee_code,
    log_type,
    camera_source,
    confidence,
    reference_time=None,
):
    initialize_attendance_logs()
    normalized_log_type = _normalize_log_type(log_type)
    log_time = _normalize_timestamp(reference_time)
    normalized_confidence = None if confidence is None else round(float(confidence), 2)

    with connect_db() as conn:
        cursor = conn.execute(
            """
            INSERT INTO attendance_logs (
                employee_code,
                log_time,
                log_type,
                camera_source,
                confidence
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                employee_code,
                log_time,
                normalized_log_type,
                camera_source,
                normalized_confidence,
            ),
        )
        new_log_id = cursor.lastrowid
        row = conn.execute(
            """
            SELECT id, employee_code, log_time, log_type, camera_source, confidence
            FROM attendance_logs
            WHERE id = ?
            """,
            (new_log_id,),
        ).fetchone()

    clear_attendance_cache()
    return _row_to_dict(row)


@st.cache_data(show_spinner=False)
def get_latest_attendance_log(employee_code=None, log_type=None):
    initialize_attendance_logs()
    query = (
        "SELECT id, employee_code, log_time, log_type, camera_source, confidence "
        "FROM attendance_logs WHERE 1 = 1"
    )
    params = []

    if employee_code:
        query += " AND employee_code = ?"
        params.append(employee_code)

    if log_type:
        query += " AND log_type = ?"
        params.append(_normalize_log_type(log_type))

    query += " ORDER BY log_time DESC, id DESC LIMIT 1"

    with connect_db() as conn:
        row = conn.execute(query, params).fetchone()

    return _row_to_dict(row)


def get_today_logs(employee_code, reference_time=None):
    initialize_attendance_logs()
    current_date = _normalize_date(reference_time)

    with connect_db() as conn:
        rows = conn.execute(
            """
            SELECT id, employee_code, log_time, log_type, camera_source, confidence
            FROM attendance_logs
            WHERE employee_code = ?
              AND date(log_time) = ?
            ORDER BY log_time ASC, id ASC
            """,
            (employee_code, current_date),
        ).fetchall()

    return [_row_to_dict(row) for row in rows]


@st.cache_data(show_spinner=False)
def get_attendance_count(log_type=None, reference_time=None):
    initialize_attendance_logs()
    current_date = _normalize_date(reference_time)
    query = "SELECT COUNT(*) FROM attendance_logs WHERE date(log_time) = ?"
    params = [current_date]

    if log_type:
        query += " AND log_type = ?"
        params.append(_normalize_log_type(log_type))

    with connect_db() as conn:
        count = conn.execute(query, params).fetchone()[0]

    return count


@st.cache_data(show_spinner=False)
def get_today_attendance_dataframe(reference_time=None):
    initialize_attendance_logs()
    current_date = _normalize_date(reference_time)

    query = (
        "SELECT id, employee_code, log_time, log_type, camera_source, confidence "
        "FROM attendance_logs WHERE date(log_time) = ? "
        "ORDER BY log_time DESC, id DESC"
    )

    with connect_db() as conn:
        return pd.read_sql(query, conn, params=[current_date])


def can_check_in(employee_code, reference_time=None):
    today_logs = get_today_logs(employee_code, reference_time)
    has_check_in_today = any(log["log_type"] == "IN" for log in today_logs)

    if not has_check_in_today:
        return True, "The employee can check in today."

    return False, "The employee has already checked in today."


def can_check_out(employee_code, reference_time=None):
    today_logs = get_today_logs(employee_code, reference_time)

    if not today_logs:
        return False, "The employee has not checked in today, so check-out is not allowed."

    latest_log = today_logs[-1]
    has_check_in_today = any(log["log_type"] == "IN" for log in today_logs)

    if not has_check_in_today:
        return False, "The employee has not checked in today, so check-out is not allowed."

    if latest_log["log_type"] == "OUT":
        return False, "The employee has already checked out today."

    return True, "The employee can check out today."


def register_check_in(employee_code, camera_source, confidence, reference_time=None):
    can_log, message = can_check_in(employee_code, reference_time)

    if not can_log:
        raise ValueError(message)

    return create_attendance_log(
        employee_code=employee_code,
        log_type="IN",
        camera_source=camera_source,
        confidence=confidence,
        reference_time=reference_time,
    )


def register_check_out(employee_code, camera_source, confidence, reference_time=None):
    can_log, message = can_check_out(employee_code, reference_time)

    if not can_log:
        raise ValueError(message)

    return create_attendance_log(
        employee_code=employee_code,
        log_type="OUT",
        camera_source=camera_source,
        confidence=confidence,
        reference_time=reference_time,
    )


@st.cache_data(show_spinner=False)
def get_attendance_history_dataframe(employee_code=None, log_type=None):
    initialize_attendance_logs()
    query = (
        "SELECT al.id, al.employee_code, e.name, al.log_time, al.log_type, "
        "al.camera_source, al.confidence "
        "FROM attendance_logs al "
        "LEFT JOIN employees e ON e.employee_code = al.employee_code "
        "WHERE 1 = 1"
    )
    params = []

    if employee_code:
        query += " AND al.employee_code = ?"
        params.append(employee_code)

    if log_type:
        query += " AND al.log_type = ?"
        params.append(_normalize_log_type(log_type))

    query += " ORDER BY al.log_time DESC, al.id DESC"

    with connect_db() as conn:
        return pd.read_sql(query, conn, params=params)



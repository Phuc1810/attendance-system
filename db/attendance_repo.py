from datetime import datetime
import pandas as pd
import streamlit as st
from sqlalchemy import select, desc, func, and_
from sqlalchemy.orm import Session

from db.session import SessionLocal, engine, init_db
from db.models import AttendanceLog, Employee

VALID_LOG_TYPES = {"IN", "OUT"}

def initialize_attendance_logs():
    init_db()

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

def _row_to_dict(log_obj):
    if not log_obj:
        return None
    return {
        "id": log_obj.id,
        "employee_code": log_obj.employee_code,
        "log_time": log_obj.log_time,
        "log_type": log_obj.log_type,
        "camera_source": log_obj.camera_source,
        "confidence": log_obj.confidence,
    }

def clear_attendance_cache():
    get_latest_attendance_log.clear()
    get_attendance_count.clear()
    get_today_attendance_dataframe.clear()
    get_attendance_history_dataframe.clear()

def create_attendance_log(
    employee_code, log_type, camera_source, confidence, reference_time=None
):
    initialize_attendance_logs()
    normalized_log_type = _normalize_log_type(log_type)
    log_time = _normalize_timestamp(reference_time)
    normalized_confidence = None if confidence is None else round(float(confidence), 2)

    with SessionLocal() as db:
        new_log = AttendanceLog(
            employee_code=employee_code,
            log_time=log_time,
            log_type=normalized_log_type,
            camera_source=camera_source,
            confidence=normalized_confidence,
        )
        db.add(new_log)
        db.commit()
        db.refresh(new_log)
        result = _row_to_dict(new_log)

    clear_attendance_cache()
    return result

@st.cache_data(show_spinner=False)
def get_latest_attendance_log(employee_code=None, log_type=None):
    initialize_attendance_logs()
    with SessionLocal() as db:
        stmt = select(AttendanceLog)
        if employee_code:
            stmt = stmt.where(AttendanceLog.employee_code == employee_code)
        if log_type:
            stmt = stmt.where(AttendanceLog.log_type == _normalize_log_type(log_type))
        
        stmt = stmt.order_by(desc(AttendanceLog.log_time), desc(AttendanceLog.id)).limit(1)
        log_obj = db.scalars(stmt).first()
        return _row_to_dict(log_obj)

def get_today_logs(employee_code, reference_time=None):
    initialize_attendance_logs()
    current_date = _normalize_date(reference_time)

    with SessionLocal() as db:
        stmt = select(AttendanceLog).where(
            and_(
                AttendanceLog.employee_code == employee_code,
                func.substr(AttendanceLog.log_time, 1, 10) == current_date
            )
        ).order_by(AttendanceLog.log_time, AttendanceLog.id)
        
        logs = db.scalars(stmt).all()
        return [_row_to_dict(log) for log in logs]

@st.cache_data(show_spinner=False)
def get_attendance_count(log_type=None, reference_time=None):
    initialize_attendance_logs()
    current_date = _normalize_date(reference_time)

    with SessionLocal() as db:
        stmt = select(func.count(AttendanceLog.id)).where(
            func.substr(AttendanceLog.log_time, 1, 10) == current_date
        )
        if log_type:
            stmt = stmt.where(AttendanceLog.log_type == _normalize_log_type(log_type))
        return db.scalar(stmt) or 0

@st.cache_data(show_spinner=False)
def get_today_attendance_dataframe(reference_time=None):
    initialize_attendance_logs()
    current_date = _normalize_date(reference_time)
    
    query = (
        "SELECT id, employee_code, log_time, log_type, camera_source, confidence "
        f"FROM attendance_logs WHERE substr(log_time, 1, 10) = '{current_date}' "
        "ORDER BY log_time DESC, id DESC"
    )
    with engine.connect() as conn:
        return pd.read_sql(query, conn)

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
    return create_attendance_log(employee_code, "IN", camera_source, confidence, reference_time)

def register_check_out(employee_code, camera_source, confidence, reference_time=None):
    can_log, message = can_check_out(employee_code, reference_time)
    if not can_log:
        raise ValueError(message)
    return create_attendance_log(employee_code, "OUT", camera_source, confidence, reference_time)

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
    
    with engine.connect() as conn:
        return pd.read_sql(query, conn, params=params)

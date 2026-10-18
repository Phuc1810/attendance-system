from pathlib import Path
import pandas as pd
import streamlit as st
from sqlalchemy.orm import Session
from sqlalchemy import select

from db.session import init_db, SessionLocal, engine
from db.models import Employee

EMPLOYEE_CODE_PREFIX = "NV"
EMPLOYEE_CODE_WIDTH = 3

def format_employee_code(number):
    return f"{EMPLOYEE_CODE_PREFIX}{number:0{EMPLOYEE_CODE_WIDTH}d}"

def parse_employee_code(employee_code):
    if not employee_code or not employee_code.startswith(EMPLOYEE_CODE_PREFIX):
        return None
    number_part = employee_code[len(EMPLOYEE_CODE_PREFIX):]
    if not number_part.isdigit():
        return None
    return int(number_part)

def _get_next_employee_code(db: Session):
    stmt = select(Employee.employee_code).where(Employee.employee_code != None)
    used_numbers = {
        parse_employee_code(code) for code in db.scalars(stmt)
        if parse_employee_code(code) is not None
    }
    next_number = 1
    while next_number in used_numbers:
        next_number += 1
    return format_employee_code(next_number)

def initialize_database():
    init_db()
    with SessionLocal() as db:
        # fetch all employees to sync folders
        employees = db.execute(select(Employee.id, Employee.employee_code).order_by(Employee.id)).fetchall()
        from core.save_face import sync_employee_face_folders
        sync_employee_face_folders(employees)

def get_employee_codes():
    initialize_database()
    with SessionLocal() as db:
        stmt = select(Employee.employee_code).where(Employee.employee_code != None).order_by(Employee.id)
        return list(db.scalars(stmt).all())

@st.cache_data(show_spinner=False)
def get_all_employees():
    with SessionLocal() as db:
        stmt = select(Employee.id, Employee.employee_code, Employee.name, Employee.department).order_by(Employee.id)
        # Returns list of tuples
        return db.execute(stmt).fetchall()

@st.cache_data(show_spinner=False)
def get_employee_dataframe():
    with engine.connect() as conn:
        return pd.read_sql(
            "SELECT employee_code AS code, name, department, id FROM employees ORDER BY id",
            conn,
        )

def clear_employee_cache():
    get_all_employees.clear()
    get_employee_dataframe.clear()

def validate_employee_fields(name, department):
    errors = []
    if not name or not name.strip():
        errors.append("Employee name is required.")
    if not department or not department.strip():
        errors.append("Department is required.")
    return errors

def create_employee(name, department):
    initialize_database()
    errors = validate_employee_fields(name, department)
    if errors:
        raise ValueError(" ".join(errors))

    with SessionLocal() as db:
        employee_code = _get_next_employee_code(db)
        new_employee = Employee(employee_code=employee_code, name=name.strip(), department=department.strip())
        db.add(new_employee)
        db.commit()

    from core.save_face import ensure_employee_folder
    ensure_employee_folder(employee_code)
    clear_employee_cache()
    return employee_code

def update_employee(employee_id, name, department):
    initialize_database()
    errors = validate_employee_fields(name, department)
    if errors:
        raise ValueError(" ".join(errors))

    with SessionLocal() as db:
        emp = db.get(Employee, employee_id)
        if not emp:
            raise ValueError("Employee not found.")
        emp.name = name.strip()
        emp.department = department.strip()
        db.commit()

    clear_employee_cache()

def delete_employee(employee_id):
    initialize_database()
    employee_code = None
    with SessionLocal() as db:
        emp = db.get(Employee, employee_id)
        if not emp:
            raise ValueError("Employee not found.")
        employee_code = emp.employee_code
        db.delete(emp)
        db.commit()

    from core.save_face import delete_employee_folder
    if employee_code:
        delete_employee_folder(employee_code)

    clear_employee_cache()
    return employee_code

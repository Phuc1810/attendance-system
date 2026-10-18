from datetime import datetime
from sqlalchemy import Column, Integer, String, Index, Float, ForeignKey
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class Employee(Base):
    __tablename__ = "employees"
    id = Column(Integer, primary_key=True, autoincrement=True)
    employee_code = Column(String, unique=True, index=True)
    name = Column(String, nullable=False)
    department = Column(String, nullable=False)

class AttendanceLog(Base):
    __tablename__ = "attendance_logs"
    id = Column(Integer, primary_key=True, autoincrement=True)
    employee_code = Column(String, ForeignKey("employees.employee_code"), nullable=False)
    log_time = Column(String, nullable=False)
    log_type = Column(String, nullable=False)
    camera_source = Column(String)
    confidence = Column(Float)

    __table_args__ = (
        Index("idx_attendance_logs_employee_time", "employee_code", "log_time"),
        Index("idx_attendance_logs_time", "log_time"),
    )

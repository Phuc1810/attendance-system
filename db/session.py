import os
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from db.models import Base

DB_PATH = Path(__file__).resolve().parent.parent / "attendance.db"
# Use sqlite backend
SQLALCHEMY_DATABASE_URL = f"sqlite:///{DB_PATH}"

# create_engine specific to sqlite (check_same_thread=False for Streamlit multithreading)
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

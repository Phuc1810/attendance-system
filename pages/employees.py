import pandas as pd
import streamlit as st

from db.database import connect_db, initialize_database

initialize_database()

st.title("Employee Management")

name = st.text_input("Employee Name")
department = st.text_input("Department")

if st.button("Add Employee"):
    if not name.strip() or not department.strip():
        st.warning("Please enter both employee name and department.")
    else:
        with connect_db() as conn:
            conn.execute(
                "INSERT INTO employees (name, department) VALUES (?, ?)",
                (name.strip(), department.strip()),
            )
        st.success("Employee added!")

st.subheader("Employee List")

with connect_db() as conn:
    df = pd.read_sql("SELECT * FROM employees ORDER BY id", conn)

st.dataframe(df)

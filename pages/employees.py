import streamlit as st
import sqlite3
import pandas as pd

conn = sqlite3.connect("attendance.db", check_same_thread=False)

st.title("Employee Management")

name = st.text_input("Employee Name")
department = st.text_input("Department")

if st.button("Add Employee"):
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO employees (name, department) VALUES (?, ?)",
        (name, department)
    )
    conn.commit()
    st.success("Employee added!")

st.subheader("Employee List")

df = pd.read_sql("SELECT * FROM employees", conn)

st.dataframe(df)

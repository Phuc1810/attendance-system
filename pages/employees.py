import streamlit as st

from db.database import create_employee, get_employee_dataframe, initialize_database

# Gọi hàm để đảm bảo file .db đã được tạo 
initialize_database()

st.title("Employee Management")

name = st.text_input("Employee Name")
department = st.text_input("Department")

if st.button("Add Employee"):
    if not name.strip() or not department.strip():
        st.warning("Please enter both employee name and department.")
    else:
        employee_code = create_employee(name, department)
        st.success(f"Employee added with code {employee_code}!")

st.subheader("Employee List")

st.dataframe(get_employee_dataframe(), use_container_width=True)

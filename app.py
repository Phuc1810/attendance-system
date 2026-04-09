import streamlit as st

from db.attendance_repo import initialize_attendance_logs
from db.database import initialize_database

initialize_database()
initialize_attendance_logs()

st.set_page_config(page_title="Attendance System")

st.sidebar.title("Attendance System")
st.sidebar.caption("Face recognition attendance workflow")

navigation = st.navigation(
    [
        st.Page("pages/1_Dashboard.py", title="Dashboard"),
        st.Page("pages/2_Employees.py", title="Employees"),
        st.Page("pages/3_Register_Face.py", title="Register Face"),
        st.Page("pages/4_Attendance.py", title="Attendance"),
        st.Page("pages/5_History.py", title="History"),
        st.Page("pages/6_Face_Detection.py", title="Face Detection"),
        st.Page("pages/7_Face_Recognition.py", title="Face Recognition"),
    ],
    position="sidebar",
)

navigation.run()

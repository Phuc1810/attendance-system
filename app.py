import streamlit as st

from db.attendance_repo import initialize_attendance_logs
from db.database import initialize_database
from ui.styles import apply_custom_css

st.set_page_config(page_title="Attendance System", layout="wide", initial_sidebar_state="expanded")
apply_custom_css()

initialize_database()
initialize_attendance_logs()

st.sidebar.title("🏢 Smart Attendance")
st.sidebar.caption("Powered by Face Recognition AI")

pages = {
    "Quản lý (Management)": [
        st.Page("pages/1_Dashboard.py", title="Dashboard", icon="📊"),
        st.Page("pages/2_Employees.py", title="Employees", icon="👥"),
        st.Page("pages/5_History.py", title="History", icon="🕒"),
    ],
    "Nhận diện (Face AI)": [
        st.Page("pages/4_Attendance.py", title="Check-in Camera", icon="📷"),
        st.Page("pages/3_Register_Face.py", title="Register Face", icon="🧑‍💻"),
    ],
    "Kiểm thử (System Testing)": [
        st.Page("pages/6_Face_Detection.py", title="Face Detection", icon="🔍"),
        st.Page("pages/7_Face_Recognition.py", title="Face Recognition", icon="🤖"),
    ],
}

navigation = st.navigation(pages, position="sidebar")
navigation.run()

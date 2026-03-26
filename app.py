import streamlit as st

from db.database import initialize_database

initialize_database()

st.set_page_config(page_title="Attendance System")

st.title("Face Recognition Attendance System")

st.write("AI Project - Computer Vision")

st.write("Use the sidebar to navigate.")

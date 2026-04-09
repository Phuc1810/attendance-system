import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from core.camera_stream import release_inactive_cameras
from db.attendance_repo import (
    get_attendance_count,
    get_latest_attendance_log,
    get_today_attendance_dataframe,
    initialize_attendance_logs,
)
from db.database import get_employee_codes, get_employee_dataframe, initialize_database

# --- CONFIG & INITIALIZATION ---
st.set_page_config(page_title="Attendance Dashboard", layout="wide")
st_autorefresh(interval=5000, key="data_refresh")

initialize_database()
initialize_attendance_logs()
release_inactive_cameras(st.session_state)

# --- SIDEBAR FILTERS ---
st.sidebar.title("Search & Filters")
employee_code_options = ["All"] + get_employee_codes()
selected_employee_code = st.sidebar.selectbox(
    "Employee Code",
    options=employee_code_options,
    help="You can type in this field to search and select an employee code from the database.",
)
filter_type = st.sidebar.multiselect(
    "Filter Type",
    options=["IN", "OUT"],
    default=["IN", "OUT"],
)

# --- DATA FETCHING ---
employees_df = get_employee_dataframe()
today_attendance_df = get_today_attendance_dataframe()
latest_log = get_latest_attendance_log()

employee_name_map = {}
if not employees_df.empty:
    employee_name_map = employees_df.set_index("code")["name"].to_dict()

# --- DATA PROCESSING ---
today_display_df = today_attendance_df.copy()
if not today_display_df.empty:
    today_display_df.insert(
        1,
        "Name",
        today_display_df["employee_code"].map(employee_name_map).fillna("Unknown"),
    )
    today_display_df = today_display_df.rename(
        columns={
            "employee_code": "Employee Code",
            "log_time": "Log Time",
            "log_type": "Type",
            "camera_source": "Camera",
            "confidence": "Confidence",
        }
    )

    if selected_employee_code != "All":
        today_display_df = today_display_df[
            today_display_df["Employee Code"] == selected_employee_code
        ]

    today_display_df = today_display_df[today_display_df["Type"].isin(filter_type)]

# --- UI RENDER ---
st.title("Attendance Dashboard")


# 1. Metrics Section
total_employees = len(employees_df.index)
total_in = get_attendance_count("IN")
total_out = get_attendance_count("OUT")

m1, m2, m3 = st.columns(3)
with m1:
    st.metric("Total Employees", total_employees)
with m2:
    st.metric(
        "Check In Today",
        total_in,
        delta=f"{total_in} present",
        delta_color="normal",
    )
with m3:
    st.metric("Check Out Today", total_out)

st.divider()

# 2. Main Content
content_col, log_col = st.columns([1.5, 1], gap="large")

with content_col:
    st.subheader("Attendance Overview")
    if not today_display_df.empty:
        chart_data = pd.DataFrame(
            {
                "Status": ["Check In", "Check Out"],
                "Count": [total_in, total_out],
            }
        )
        st.bar_chart(chart_data.set_index("Status"), height=200)

    st.subheader("Today Attendance List")
    if today_display_df.empty:
        st.info("No records found.")
    else:
        def color_type(value):
            color = "#2ecc71" if value == "IN" else "#e67e22"
            return f"color: {color}; font-weight: bold"

        styled_df = today_display_df.style.map(color_type, subset=["Type"])
        st.dataframe(styled_df, use_container_width=True, height=400)

with log_col:
    with st.container(border=True):
        st.subheader("Latest Log")
        if latest_log is None:
            st.info("Waiting for data...")
        else:
            emp_name = employee_name_map.get(latest_log["employee_code"], "Unknown")
            confidence = latest_log.get("confidence")
            confidence_text = "N/A" if confidence is None else f"{confidence:.2f}"
            type_text = "IN" if latest_log["log_type"] == "IN" else "OUT"

            st.markdown(
                f"""
                <div style="background-color: #0e1117; padding: 20px; border-radius: 10px; border: 1px solid #30363d">
                    <h2 style="margin:0; color: #58a6ff;">{latest_log['employee_code']}</h2>
                    <h4 style="margin:0; color: #8b949e;">{emp_name}</h4>
                    <hr style="margin: 15px 0; border: 0.5px solid #30363d">
                    <p><b>Type:</b> {type_text}</p>
                    <p><b>Camera:</b> {latest_log['camera_source']}</p>
                    <p><b>Time:</b> {latest_log['log_time']}</p>
                    <p><b>Confidence:</b> <span style="color: #238636">{confidence_text}</span></p>
                </div>
                """,
                unsafe_allow_html=True,
            )
         


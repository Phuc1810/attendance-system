import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import plotly.express as px
from datetime import datetime, timedelta

from core.camera_stream import release_inactive_cameras
from db.attendance_repo import (
    get_attendance_count,
    get_latest_attendance_log,
    get_today_attendance_dataframe,
    get_attendance_history_dataframe,
    initialize_attendance_logs,
)
from db.database import get_employee_codes, get_employee_dataframe, initialize_database

# st.set_page_config(page_title="Attendance Dashboard", layout="wide")
st_autorefresh(interval=5000, key="data_refresh")

initialize_database()
initialize_attendance_logs()
release_inactive_cameras(st.session_state)

# --- SIDEBAR FILTERS ---
st.sidebar.title("🔍 Search & Filters")
employee_code_options = ["All"] + get_employee_codes()
selected_employee_code = st.sidebar.selectbox(
    "Employee Code",
    options=employee_code_options,
    help="Select an employee code to filter the dashboard.",
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
history_df = get_attendance_history_dataframe()

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

# Plotly data processing for 7 days
last_7_days = [(datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(6, -1, -1)]
trend_data = {"Date": [], "Type": [], "Count": []}

if not history_df.empty:
    history_df["Date"] = history_df["log_time"].str[:10]
    
    for day in last_7_days:
        day_df = history_df[history_df["Date"] == day]
        in_count = len(day_df[day_df["log_type"] == "IN"])
        out_count = len(day_df[day_df["log_type"] == "OUT"])
        
        trend_data["Date"].extend([day, day])
        trend_data["Type"].extend(["Check In (IN)", "Check Out (OUT)"])
        trend_data["Count"].extend([in_count, out_count])
else:
    for day in last_7_days:
        trend_data["Date"].extend([day, day])
        trend_data["Type"].extend(["Check In (IN)", "Check Out (OUT)"])
        trend_data["Count"].extend([0, 0])

trend_df = pd.DataFrame(trend_data)


# --- UI RENDER ---
st.title("📊 Attendance Dashboard")
st.markdown("Command Center for Face Recognition Attendance System")

# 1. Metrics Section
total_employees = len(employees_df.index)
total_in = get_attendance_count("IN")
total_out = get_attendance_count("OUT")

m1, m2, m3 = st.columns(3)
with m1:
    st.metric("Total Employees 👥", total_employees)
with m2:
    st.metric(
        "Check In Today 📥",
        total_in,
        delta=f"{total_in} present",
        delta_color="normal",
    )
with m3:
    st.metric("Check Out Today 📤", total_out)

st.markdown("<br>", unsafe_allow_html=True)

# 2. Main Content
content_col, log_col = st.columns([1.8, 1], gap="large")

with content_col:
    with st.container(border=True):
        st.subheader("📈 7-Day Attendance Trend")
        fig = px.bar(
            trend_df, 
            x="Date", 
            y="Count", 
            color="Type", 
            barmode="group",
            color_discrete_map={"Check In (IN)": "#2ecc71", "Check Out (OUT)": "#e67e22"},
            labels={"Count": "Number of Logs"}
        )
        fig.update_layout(
            margin=dict(l=20, r=20, t=20, b=20),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            legend_title_text=None,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)

    with st.container(border=True):
        st.subheader("📋 Today's Attendance List")
        if today_display_df.empty:
            st.info("No records found.")
        else:
            st.dataframe(
                today_display_df,
                use_container_width=True, 
                height=300,
                hide_index=True,
                column_config={
                    "Confidence": st.column_config.ProgressColumn(
                        "Confidence",
                        help="Face recognition confidence",
                        format="%.2f",
                        min_value=0,
                        max_value=1.0,
                    ),
                }
            )

with log_col:
    with st.container(border=True):
        st.subheader("🔔 Latest Log")
        if latest_log is None:
            st.info("Waiting for data...")
        else:
            emp_name = employee_name_map.get(latest_log["employee_code"], "Unknown")
            confidence = latest_log.get("confidence")
            confidence_text = "N/A" if confidence is None else f"{confidence:.2f}"
            type_text = "IN" if latest_log["log_type"] == "IN" else "OUT"
            type_color = "#2ecc71" if type_text == "IN" else "#e67e22"

            st.markdown(
                f"""
                <div style="background-color: #FFFFFF; padding: 20px; border-radius: 8px; border: 1px solid #E2E8F0; box-shadow: 0 1px 3px rgba(0,0,0,0.1)">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <h2 style="margin:0; color: #1E3A8A;">{latest_log['employee_code']}</h2>
                        <span style="background-color: {type_color}; color: white; padding: 4px 10px; border-radius: 20px; font-weight: bold; font-size: 14px;">{type_text}</span>
                    </div>
                    <h4 style="margin:5px 0 0 0; color: #64748B;">{emp_name}</h4>
                    <hr style="margin: 15px 0; border: 0.5px solid #E2E8F0">
                    <p style="margin: 5px 0; color: #334155;"><b>Camera:</b> {latest_log['camera_source']}</p>
                    <p style="margin: 5px 0; color: #334155;"><b>Time:</b> {latest_log['log_time']}</p>
                    <p style="margin: 5px 0; color: #334155;"><b>Confidence:</b> <span style="color: #10B981; font-weight: bold;">{confidence_text}</span></p>
                </div>
                """,
                unsafe_allow_html=True,
            )

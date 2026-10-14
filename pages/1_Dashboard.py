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

# Tính toán các chỉ số bổ trợ & tỷ lệ phần trăm
attendance_rate = (total_in / total_employees * 100) if total_employees > 0 else 0

# Tinh chỉnh delta & huy hiệu thân thiện cho trạng thái trống (Empty State số 0)
if total_in == 0:
    in_delta_text = "⚪ Chưa có lượt vào hôm nay"
    in_delta_color = "off"
else:
    in_delta_text = f"🟢 {total_in}/{total_employees} có mặt ({attendance_rate:.0f}%)"
    in_delta_color = "normal"

if total_out == 0:
    out_delta_text = "⚪ Chưa có lượt ra hôm nay"
    out_delta_color = "off"
else:
    out_delta_text = f"🟠 {total_out} lượt ra đã ghi nhận"
    out_delta_color = "normal"

m1, m2, m3 = st.columns(3)
with m1:
    st.metric(
        label="Total Employees 👥",
        value=total_employees,
        delta=f"{total_employees} hồ sơ nhân sự",
        delta_color="off",
    )
with m2:
    st.metric(
        label="Check In Today 📥",
        value=total_in,
        delta=in_delta_text,
        delta_color=in_delta_color,
    )
with m3:
    st.metric(
        label="Check Out Today 📤",
        value=total_out,
        delta=out_delta_text,
        delta_color=out_delta_color,
    )

st.markdown("<br>", unsafe_allow_html=True)

# 2. Main Content: Grid 2:1 cân xứng giữa Biểu đồ xu hướng và Latest Log
content_col, log_col = st.columns([2, 1], gap="large")

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
            margin=dict(l=15, r=15, t=15, b=15),
            height=320,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            legend_title_text=None,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)

with log_col:
    with st.container(border=True):
        st.subheader("🔔 Latest Activity")
        if latest_log is None:
            st.markdown(
                """
                <div style="background-color: #FFFFFF; height: 320px; display: flex; flex-direction: column; justify-content: center; align-items: center; border-radius: 12px; border: 1.5px dashed #CBD5E1; text-align: center; padding: 25px;">
                    <div style="font-size: 2.8rem; margin-bottom: 12px;">📡</div>
                    <h4 style="margin: 0; color: #1E3A8A; font-weight: 600;">Sẵn sàng ghi nhận</h4>
                    <p style="margin: 8px 0 0 0; color: #64748B; font-size: 0.88rem; line-height: 1.4;">
                        Chưa có lượt điểm danh nào trong phiên làm việc.<br>Dữ liệu sẽ cập nhật tự động thời gian thực.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            emp_name = employee_name_map.get(latest_log["employee_code"], "Unknown")
            confidence = latest_log.get("confidence")
            confidence_text = "N/A" if confidence is None else f"{confidence:.2f}"
            type_text = "Check In (IN)" if latest_log["log_type"] == "IN" else "Check Out (OUT)"
            type_color = "#10B981" if latest_log["log_type"] == "IN" else "#F59E0B"
            type_bg = "#ECFDF5" if latest_log["log_type"] == "IN" else "#FFFBEB"

            st.markdown(
                f"""
                <div style="background-color: #FFFFFF; height: 320px; display: flex; flex-direction: column; justify-content: space-between; padding: 24px; border-radius: 12px; border: 1px solid #E2E8F0; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);">
                    <div>
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                            <span style="font-size: 1.35rem; font-weight: 700; color: #1E3A8A;">{latest_log['employee_code']}</span>
                            <span style="background-color: {type_bg}; color: {type_color}; border: 1px solid {type_color}; padding: 4px 12px; border-radius: 20px; font-weight: 600; font-size: 0.82rem;">{type_text}</span>
                        </div>
                        <h4 style="margin: 0; color: #334155; font-weight: 600;">{emp_name}</h4>
                        <hr style="margin: 16px 0; border: none; border-top: 1px solid #F1F5F9;">
                    </div>
                    <div style="font-size: 0.9rem; color: #475569; display: flex; flex-direction: column; gap: 8px;">
                        <div style="display: flex; justify-content: space-between;">
                            <span style="color: #64748B;">Cổng Camera:</span>
                            <span style="font-weight: 600; color: #0F172A;">{latest_log['camera_source']}</span>
                        </div>
                        <div style="display: flex; justify-content: space-between;">
                            <span style="color: #64748B;">Thời gian:</span>
                            <span style="font-weight: 600; color: #0F172A;">{latest_log['log_time']}</span>
                        </div>
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span style="color: #64748B;">Độ tin cậy AI:</span>
                            <span style="background-color: #ECFDF5; color: #059669; font-weight: 700; padding: 2px 8px; border-radius: 6px; font-size: 0.85rem;">{confidence_text}</span>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

# 3. Today's Attendance List (Full Width bên dưới, thoáng đãng và tích hợp bộ lọc trực tiếp)
with st.container(border=True):
    st.subheader("📋 Today's Attendance List")
    
    # In-page Filter Bar: Lọc trực tiếp ngay trên bảng dữ liệu, không làm rối sidebar
    employee_code_options = ["All"] + get_employee_codes()
    filter_col_emp, filter_col_type = st.columns([1.2, 1])
    with filter_col_emp:
        selected_employee_code = st.selectbox(
            "Filter by Employee",
            options=employee_code_options,
            label_visibility="collapsed",
            help="Filter today attendance by specific employee",
        )
    with filter_col_type:
        filter_type = st.multiselect(
            "Filter Type",
            options=["IN", "OUT"],
            default=["IN", "OUT"],
            label_visibility="collapsed",
            placeholder="Log Type (IN/OUT)",
        )

    filtered_today_df = today_display_df.copy()
    if not filtered_today_df.empty:
        if selected_employee_code != "All":
            filtered_today_df = filtered_today_df[
                filtered_today_df["Employee Code"] == selected_employee_code
            ]
        if filter_type:
            filtered_today_df = filtered_today_df[filtered_today_df["Type"].isin(filter_type)]

    if filtered_today_df.empty:
        st.info("No matching records found.")
    else:
        st.dataframe(
            filtered_today_df,
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

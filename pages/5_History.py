from datetime import datetime
from io import BytesIO

import pandas as pd
import streamlit as st
from core.camera_stream import release_inactive_cameras

from db.attendance_repo import get_attendance_history_dataframe, initialize_attendance_logs
from db.database import get_employee_codes, initialize_database

initialize_database()
initialize_attendance_logs()
st.set_page_config(page_title="Attendance Dashboard", layout="wide")
release_inactive_cameras(st.session_state)
st.title("Attendance History")
st.caption("Review attendance records, filter the data you need, and export the current view.")


def build_export_file_name(extension, employee_code, log_type):
    employee_part = "all" if employee_code == "All" else employee_code.lower()
    log_type_part = "all" if log_type == "All" else log_type.lower()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"attendance_history_{employee_part}_{log_type_part}_{timestamp}.{extension}"


def dataframe_to_excel_bytes(dataframe):
    output = BytesIO()

    try:
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            dataframe.to_excel(writer, index=False, sheet_name="Attendance History")
    except ImportError:
        return None

    output.seek(0)
    return output.getvalue()


def format_history_dataframe(dataframe):
    if dataframe.empty:
        return dataframe

    display_df = dataframe.copy()
    display_df["confidence"] = display_df["confidence"].round(2)
    display_df = display_df.rename(
        columns={
            "id": "ID",
            "employee_code": "Employee Code",
            "name": "Name",
            "log_time": "Log Time",
            "log_type": "Type",
            "camera_source": "Camera",
            "confidence": "Confidence",
        }
    )

    return display_df[
        [
            "ID",
            "Employee Code",
            "Name",
            "Type",
            "Log Time",
            "Camera",
            "Confidence",
        ]
    ]


employee_options = ["All"] + get_employee_codes()
log_type_options = ["All", "IN", "OUT"]

filter_col, export_col = st.columns([1.2, 1], gap="large")

with filter_col:
    with st.container(border=True):
        st.subheader("Filter Data")
        st.caption("Use filters to narrow down the records before exporting or reviewing the table.")

        filter_input_col_1, filter_input_col_2 = st.columns(2)
        with filter_input_col_1:
            selected_employee_code = st.selectbox(
                "Employee Code",
                employee_options,
            )
        with filter_input_col_2:
            selected_log_type = st.selectbox(
                "Log Type",
                log_type_options,
            )

employee_filter = None if selected_employee_code == "All" else selected_employee_code
log_type_filter = None if selected_log_type == "All" else selected_log_type

history_df = get_attendance_history_dataframe(
    employee_code=employee_filter,
    log_type=log_type_filter,
)
display_history_df = format_history_dataframe(history_df)

total_records = len(history_df.index)
check_in_count = int((history_df["log_type"] == "IN").sum()) if not history_df.empty else 0
check_out_count = int((history_df["log_type"] == "OUT").sum()) if not history_df.empty else 0

with export_col:
    with st.container(border=True):
        st.subheader("Export Data")
        st.caption("Export the exact dataset currently shown in the table below as Excel.")

        metric_col_1, metric_col_2, metric_col_3 = st.columns(3)
        metric_col_1.metric("Records", total_records)
        metric_col_2.metric("IN", check_in_count)
        metric_col_3.metric("OUT", check_out_count)

        excel_bytes = dataframe_to_excel_bytes(display_history_df)
        if excel_bytes is None:
            st.info("Install openpyxl if you want Excel export.")
        else:
            st.download_button(
                "Export Excel",
                data=excel_bytes,
                file_name=build_export_file_name(
                    "xlsx",
                    selected_employee_code,
                    selected_log_type,
                ),
                mime=(
                    "application/vnd.openxmlformats-officedocument."
                    "spreadsheetml.sheet"
                ),
                use_container_width=True,
                disabled=display_history_df.empty,
            )

with st.container(border=True):
    st.subheader("Attendance Data")
    st.caption("Newest attendance records are shown first.")

    if display_history_df.empty:
        st.info("No attendance history found for the selected filters.")
    else:
        st.dataframe(display_history_df, use_container_width=True)

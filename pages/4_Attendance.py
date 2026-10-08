from datetime import datetime, timedelta
import math

import cv2
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from streamlit_autorefresh import st_autorefresh

from core.camera_stream import AttendanceProcessor
from db.attendance_repo import (
    get_latest_attendance_log,
    initialize_attendance_logs,
    register_check_in,
    register_check_out,
)
from db.database import initialize_database

# st.set_page_config(page_title="Attendance Dashboard", layout="wide")
st_autorefresh(interval=1000, key="data_refresh")

PAGE_KEY = "attendance"
AUTO_MATCH_REQUIRED_FRAMES = 15
AUTO_ACTION_COOLDOWN_SECONDS = 30
AUTO_RETRY_COOLDOWN_SECONDS = 15
NOTICE_DURATION_SECONDS = 5

CAMERA_CONFIGS = {
    0: {
        "title": "Laptop Camera - Check In",
        "camera_source": "Laptop Camera",
        "log_type": "IN",
        "description": "Camera 0 is dedicated to automatic check-in.",
    },
    1: {
        "title": "Rappo C200 - Check Out",
        "camera_source": "Rappo C200",
        "log_type": "OUT",
        "description": "Camera 1 is dedicated to automatic check-out.",
    },
}

NOTICE_ICONS = {
    "success": "✅",
    "warning": "⚠️",
    "info": "ℹ️",
}

initialize_database()
initialize_attendance_logs()

st.title("Attendance")

def get_camera_config(camera_index):
    return CAMERA_CONFIGS.get(camera_index, CAMERA_CONFIGS[0])

def get_action_text(camera_config):
    return "check in" if camera_config["log_type"] == "IN" else "check out"

def create_attendance_from_camera(employee_code, confidence, camera_index):
    camera_config = get_camera_config(camera_index)
    if camera_config["log_type"] == "IN":
        return register_check_in(
            employee_code=employee_code,
            camera_source=camera_config["camera_source"],
            confidence=confidence,
        )
    return register_check_out(
        employee_code=employee_code,
        camera_source=camera_config["camera_source"],
        confidence=confidence,
    )

def set_attendance_notice(level, title, message, duration_seconds=NOTICE_DURATION_SECONDS):
    st.session_state[f"{PAGE_KEY}_notice"] = {
        "level": level,
        "title": title,
        "message": message,
        "expires_at": (datetime.now() + timedelta(seconds=duration_seconds)).isoformat(),
    }

def get_active_notice():
    notice = st.session_state.get(f"{PAGE_KEY}_notice")
    if not notice:
        return None
    expires_at = notice.get("expires_at")
    if not expires_at:
        return notice
    if datetime.now() >= datetime.fromisoformat(expires_at):
        st.session_state.pop(f"{PAGE_KEY}_notice", None)
        return None
    return notice

def render_attendance_notice():
    notice = get_active_notice()
    if not notice:
        return
    icon = NOTICE_ICONS.get(notice["level"], NOTICE_ICONS["info"])
    st.toast(f"{notice['title']}: {notice['message']}", icon=icon)
    st.session_state.pop(f"{PAGE_KEY}_notice", None)

def get_recent_attempt_key(employee_code, camera_config):
    return f"{employee_code}:{camera_config['log_type']}"

def get_recent_attempt_remaining(employee_code, camera_config):
    attempt_key = st.session_state.get(f"{PAGE_KEY}_last_attempt_key")
    attempt_time_raw = st.session_state.get(f"{PAGE_KEY}_last_attempt_at")
    current_key = get_recent_attempt_key(employee_code, camera_config)
    if attempt_key != current_key or not attempt_time_raw:
        return 0
    attempt_time = datetime.fromisoformat(attempt_time_raw)
    remaining_seconds = AUTO_RETRY_COOLDOWN_SECONDS - (datetime.now() - attempt_time).total_seconds()
    return max(0, math.ceil(remaining_seconds))

def mark_recent_attempt(employee_code, camera_config):
    st.session_state[f"{PAGE_KEY}_last_attempt_key"] = get_recent_attempt_key(employee_code, camera_config)
    st.session_state[f"{PAGE_KEY}_last_attempt_at"] = datetime.now().isoformat()

def get_log_cooldown_remaining(employee_code, camera_config):
    latest_log = get_latest_attendance_log(employee_code)
    if not latest_log:
        return 0
    if latest_log["log_type"] != camera_config["log_type"]:
        return 0
    log_time = datetime.strptime(latest_log["log_time"], "%Y-%m-%d %H:%M:%S")
    remaining_seconds = AUTO_ACTION_COOLDOWN_SECONDS - (datetime.now() - log_time).total_seconds()
    return max(0, math.ceil(remaining_seconds))


control_col_1, control_col_3 = st.columns([1.2, 1.4], gap="large")
with control_col_1:
    selected_camera_index = st.selectbox(
        "Choose camera",
        [0, 1],
        format_func=lambda index: get_camera_config(index)["title"],
        key=f"{PAGE_KEY}_camera_index",
    )

selected_camera_config = get_camera_config(selected_camera_index)

with control_col_3:
    with st.container(border=True):
        st.caption("Auto Attendance Flow")
        st.markdown(f"**{selected_camera_config['title']}**")
        st.write(selected_camera_config["description"])
        st.caption(f"Requirement: {AUTO_MATCH_REQUIRED_FRAMES} stable frames before the system auto logs attendance.")

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

render_attendance_notice()
preview_col, details_col = st.columns([1.8, 0.95], gap="large")

with preview_col:
    with st.container(border=True):
        st.subheader("Live Camera")
        st.caption("Keep exactly one face inside the frame. When recognition stays stable long enough, attendance is recorded automatically.")
        
        ctx = webrtc_streamer(
            key="attendance",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_processor_factory=AttendanceProcessor,
            media_stream_constraints={
                "video": {
                    "width": {"ideal": 640},
                    "height": {"ideal": 480}
                },
                "audio": False
            },
            async_processing=True,
        )

        if ctx and ctx.video_processor:
            ctx.video_processor.camera_index = selected_camera_index

with details_col:
    recognized_employee_code = None
    recognized_confidence = None
    recognized_threshold = None
    stable_count = 0
    
    if ctx and ctx.state.playing and ctx.video_processor:
        recognized_employee_code = ctx.video_processor.recognized_employee_code
        recognized_confidence = ctx.video_processor.recognized_confidence
        recognized_threshold = ctx.video_processor.recognized_threshold
        stable_count = ctx.video_processor.stable_frame_count

    display_stable_count = min(stable_count, AUTO_MATCH_REQUIRED_FRAMES)
    cooldown_remaining = 0

    if recognized_employee_code:
        cooldown_remaining = get_log_cooldown_remaining(recognized_employee_code, selected_camera_config)

        if stable_count >= AUTO_MATCH_REQUIRED_FRAMES:
            retry_remaining = get_recent_attempt_remaining(recognized_employee_code, selected_camera_config)
            if retry_remaining == 0:
                mark_recent_attempt(recognized_employee_code, selected_camera_config)
                action_text = get_action_text(selected_camera_config)
                
                if cooldown_remaining > 0:
                    set_attendance_notice(
                        "warning",
                        "Cooldown Active",
                        f"A recent {action_text} for {recognized_employee_code} was just recorded. Please wait {cooldown_remaining} more second(s)."
                    )
                    ctx.video_processor.stable_frame_count = 0 # reset
                else:
                    try:
                        new_log = create_attendance_from_camera(
                            employee_code=recognized_employee_code,
                            confidence=recognized_confidence,
                            camera_index=selected_camera_index,
                        )
                        set_attendance_notice(
                            "success",
                            "Attendance Recorded",
                            f"{new_log['employee_code']} {action_text} successful at {new_log['log_time']}."
                        )
                    except ValueError as error:
                        set_attendance_notice("warning", "Attendance Blocked", str(error))
                    finally:
                        ctx.video_processor.stable_frame_count = 0 # reset

    with st.container(border=True):
        st.subheader("Recognition Result")

        if recognized_employee_code:
            code_col, confidence_col = st.columns([1.1, 1])
            with code_col:
                st.caption("Employee Code")
                st.markdown(f"### {recognized_employee_code}")
            with confidence_col:
                st.metric("Confidence", f"{recognized_confidence:.2f}")

            threshold_text = "N/A" if recognized_threshold is None else f"{recognized_threshold:.2f}"
            st.write(f"**Current threshold:** {threshold_text}")

            latest_log = get_latest_attendance_log(
                recognized_employee_code,
                log_type=selected_camera_config["log_type"],
            )
            if latest_log:
                st.caption(f"Latest {get_action_text(selected_camera_config)} event for this employee")
                log_col_1, log_col_2 = st.columns(2)
                log_col_1.write(f"**Type:** {latest_log['log_type']}")
                log_col_2.write(f"**Camera:** {latest_log['camera_source']}")
                st.write(f"**Time:** {latest_log['log_time']}")
            else:
                st.caption(f"No previous {get_action_text(selected_camera_config)} record found for this employee.")
        else:
            st.info("No valid recognition result yet. Keep one face centered in the frame.")

    with st.container(border=True):
        st.subheader("Auto Attendance Mode")
        st.caption(selected_camera_config["description"])
        st.write(f"**Current mode:** {selected_camera_config['title']}")

        if not (ctx and ctx.state.playing):
            st.info("Turn on 'START' to start touchless attendance.")
        elif not recognized_employee_code:
            st.info("Waiting for one valid face so the system can start stabilizing recognition.")
            st.progress(0.0, text=f"Stable recognition progress: 0 / {AUTO_MATCH_REQUIRED_FRAMES} frames")
        else:
            progress_value = min(stable_count / AUTO_MATCH_REQUIRED_FRAMES, 1.0)
            st.progress(progress_value, text=f"Stable recognition progress: {display_stable_count} / {AUTO_MATCH_REQUIRED_FRAMES} frames")

            if cooldown_remaining > 0:
                st.warning(f"Cooldown active for {recognized_employee_code}. Please wait {cooldown_remaining} second(s).")
            elif stable_count < AUTO_MATCH_REQUIRED_FRAMES:
                remaining_frames = AUTO_MATCH_REQUIRED_FRAMES - stable_count
                st.info(f"Hold still for about {remaining_frames} stable frame(s) to trigger automatic attendance.")
            else:
                st.success("Stable recognition confirmed. Recording attendance automatically...")

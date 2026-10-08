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

st_autorefresh(interval=1000, key="data_refresh")

PAGE_KEY = "attendance"
AUTO_MATCH_REQUIRED_FRAMES = 15
AUTO_ACTION_COOLDOWN_SECONDS = 30
AUTO_RETRY_COOLDOWN_SECONDS = 15
NOTICE_DURATION_SECONDS = 5

CAMERA_CONFIGS = {
    0: {
        "title": "Camera 1 (IN) 📥",
        "camera_source": "Camera 1",
        "log_type": "IN",
        "description": "Camera 1 is dedicated to automatic check-in.",
    },
    1: {
        "title": "Camera 2 (OUT) 📤",
        "camera_source": "Camera 2",
        "log_type": "OUT",
        "description": "Camera 2 is dedicated to automatic check-out.",
    },
}

NOTICE_ICONS = {
    "success": "✅",
    "warning": "⚠️",
    "info": "ℹ️",
}

initialize_database()
initialize_attendance_logs()

st.title("📷 Face Attendance Camera")
st.markdown("Automatic Check-in/Check-out via face recognition. Select a camera mode and stand in front of the lens.")

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
    st.toast(f"**{notice['title']}**: {notice['message']}", icon=icon)
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


# Mode Selection
col_mode, col_info = st.columns([1, 2], gap="large")
with col_mode:
    selected_camera_index = st.selectbox(
        "Mode Select",
        [0, 1],
        format_func=lambda index: get_camera_config(index)["title"],
        key=f"{PAGE_KEY}_camera_index",
    )

selected_camera_config = get_camera_config(selected_camera_index)
with col_info:
    st.info(f"**Active Mode:** {selected_camera_config['title']} - {selected_camera_config['description']}")

st.divider()

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

render_attendance_notice()

preview_col, details_col = st.columns([1.5, 1], gap="large")

with preview_col:
    st.markdown("### Live Preview")
    
    # Styled container for WebRTC
    st.markdown(
        """
        <style>
        .stVideo { border-radius: 12px; overflow: hidden; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); }
        </style>
        """, 
        unsafe_allow_html=True
    )
    
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
    st.markdown("### Recognition Status")
    
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
                        f"Just recorded {action_text}. Please wait {cooldown_remaining}s."
                    )
                    ctx.video_processor.stable_frame_count = 0 
                else:
                    try:
                        new_log = create_attendance_from_camera(
                            employee_code=recognized_employee_code,
                            confidence=recognized_confidence,
                            camera_index=selected_camera_index,
                        )
                        set_attendance_notice(
                            "success",
                            "Attendance Logged",
                            f"{new_log['employee_code']} checked in successfully."
                        )
                    except ValueError as error:
                        set_attendance_notice("warning", "Skipped", str(error))
                    finally:
                        ctx.video_processor.stable_frame_count = 0 

    # Progress and Status UI
    with st.container(border=True):
        if not (ctx and ctx.state.playing):
            st.info("🟢 Turn on 'START' to begin touchless attendance.")
        elif not recognized_employee_code:
            st.markdown(
                """
                <div style="text-align: center; padding: 20px;">
                    <h3 style="color: #64748B;">Looking for face...</h3>
                    <p style="color: #94A3B8;">Please step into the frame</p>
                </div>
                """, unsafe_allow_html=True
            )
        else:
            progress_value = min(stable_count / AUTO_MATCH_REQUIRED_FRAMES, 1.0)
            
            if cooldown_remaining > 0:
                st.warning(f"⏳ Cooldown active. Wait {cooldown_remaining}s.")
            elif stable_count < AUTO_MATCH_REQUIRED_FRAMES:
                st.markdown(f"**Target Locked: {recognized_employee_code}**")
                st.progress(progress_value, text=f"Stabilizing... ({display_stable_count}/{AUTO_MATCH_REQUIRED_FRAMES})")
            else:
                st.success("✅ Stable recognition! Logging attendance...")

    # Latest Log Info
    if recognized_employee_code:
        st.markdown(
            f"""
            <div style="background-color: #FFFFFF; padding: 15px; border-radius: 8px; border: 1px solid #10B981; margin-top: 15px;">
                <h4 style="margin:0; color: #10B981;">{recognized_employee_code}</h4>
                <p style="margin: 5px 0 0 0;">Confidence: {recognized_confidence:.2f}</p>
            </div>
            """, unsafe_allow_html=True
        )

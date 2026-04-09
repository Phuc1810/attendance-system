from datetime import datetime, timedelta
import math

import cv2
import streamlit as st

st.set_page_config(page_title="Attendance Dashboard", layout="wide")

from core.camera_stream import (
    get_or_create_camera,
    get_or_update_prediction,
    release_camera,
    release_inactive_cameras,
    update_detected_faces,
)
from core.face_detect import detect_faces
from core.face_recognizer import predict_face
from core.save_face import crop_and_resize_face
from db.attendance_repo import (
    get_latest_attendance_log,
    initialize_attendance_logs,
    register_check_in,
    register_check_out,
)
from db.database import initialize_database

PAGE_KEY = "attendance"
CAMERA_INTERVAL_SECONDS = 0.15
RECOGNITION_PADDING_RATIO = 0.0
AUTO_MATCH_REQUIRED_FRAMES = 8
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

if f"{PAGE_KEY}_run_camera" not in st.session_state:
    st.session_state[f"{PAGE_KEY}_run_camera"] = False

release_inactive_cameras(st.session_state, PAGE_KEY)

camera_run_every = (
    CAMERA_INTERVAL_SECONDS
    if st.session_state.get(f"{PAGE_KEY}_run_camera", False)
    else None
)

st.title("Attendance")



def clear_recognized_employee_state():
    st.session_state[f"{PAGE_KEY}_recognized_employee_code"] = None
    st.session_state[f"{PAGE_KEY}_recognized_confidence"] = None
    st.session_state[f"{PAGE_KEY}_recognized_threshold"] = None


def reset_stability_state():
    st.session_state[f"{PAGE_KEY}_stable_employee_code"] = None
    st.session_state[f"{PAGE_KEY}_stable_frame_count"] = 0


def clear_attendance_notice():
    st.session_state.pop(f"{PAGE_KEY}_notice", None)


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
        "expires_at": (
            datetime.now() + timedelta(seconds=duration_seconds)
        ).isoformat(),
    }


def get_active_notice():
    notice = st.session_state.get(f"{PAGE_KEY}_notice")
    if not notice:
        return None

    expires_at = notice.get("expires_at")
    if not expires_at:
        return notice

    if datetime.now() >= datetime.fromisoformat(expires_at):
        clear_attendance_notice()
        return None

    return notice


def render_attendance_notice():
    notice = get_active_notice()
    if not notice:
        return

    icon = NOTICE_ICONS.get(notice["level"], NOTICE_ICONS["info"])
    st.toast(f"{notice['title']}: {notice['message']}", icon=icon)
    clear_attendance_notice()


def get_recent_attempt_key(employee_code, camera_config):
    return f"{employee_code}:{camera_config['log_type']}"


def get_recent_attempt_remaining(employee_code, camera_config):
    attempt_key = st.session_state.get(f"{PAGE_KEY}_last_attempt_key")
    attempt_time_raw = st.session_state.get(f"{PAGE_KEY}_last_attempt_at")
    current_key = get_recent_attempt_key(employee_code, camera_config)

    if attempt_key != current_key or not attempt_time_raw:
        return 0

    attempt_time = datetime.fromisoformat(attempt_time_raw)
    remaining_seconds = AUTO_RETRY_COOLDOWN_SECONDS - (
        datetime.now() - attempt_time
    ).total_seconds()
    return max(0, math.ceil(remaining_seconds))


def mark_recent_attempt(employee_code, camera_config):
    st.session_state[f"{PAGE_KEY}_last_attempt_key"] = get_recent_attempt_key(
        employee_code,
        camera_config,
    )
    st.session_state[f"{PAGE_KEY}_last_attempt_at"] = datetime.now().isoformat()


def get_log_cooldown_remaining(employee_code, camera_config):
    latest_log = get_latest_attendance_log(employee_code)
    if not latest_log:
        return 0

    if latest_log["log_type"] != camera_config["log_type"]:
        return 0

    log_time = datetime.strptime(latest_log["log_time"], "%Y-%m-%d %H:%M:%S")
    remaining_seconds = AUTO_ACTION_COOLDOWN_SECONDS - (
        datetime.now() - log_time
    ).total_seconds()
    return max(0, math.ceil(remaining_seconds))


def update_stability_state(recognized_match):
    recognized_code = recognized_match["display_code"]
    previous_code = st.session_state.get(f"{PAGE_KEY}_stable_employee_code")
    previous_count = st.session_state.get(f"{PAGE_KEY}_stable_frame_count", 0)

    if previous_code == recognized_code:
        stable_count = previous_count + 1
    else:
        stable_count = 1

    st.session_state[f"{PAGE_KEY}_stable_employee_code"] = recognized_code
    st.session_state[f"{PAGE_KEY}_stable_frame_count"] = stable_count
    return stable_count


def clear_tracking_state():
    clear_recognized_employee_state()
    reset_stability_state()


def attempt_auto_attendance(camera_config):
    recognized_code = st.session_state.get(f"{PAGE_KEY}_recognized_employee_code")
    recognized_confidence = st.session_state.get(f"{PAGE_KEY}_recognized_confidence")
    stable_count = st.session_state.get(f"{PAGE_KEY}_stable_frame_count", 0)

    if not recognized_code or stable_count < AUTO_MATCH_REQUIRED_FRAMES:
        return

    retry_remaining = get_recent_attempt_remaining(recognized_code, camera_config)
    if retry_remaining > 0:
        return

    mark_recent_attempt(recognized_code, camera_config)

    cooldown_remaining = get_log_cooldown_remaining(recognized_code, camera_config)
    action_text = get_action_text(camera_config)

    if cooldown_remaining > 0:
        set_attendance_notice(
            "warning",
            "Cooldown Active",
            (
                f"A recent {action_text} for {recognized_code} was just recorded. "
                f"Please wait {cooldown_remaining} more second(s)."
            ),
        )
        reset_stability_state()
        return

    try:
        new_log = create_attendance_from_camera(
            employee_code=recognized_code,
            confidence=recognized_confidence,
            camera_index=st.session_state[f"{PAGE_KEY}_camera_index"],
        )
        set_attendance_notice(
            "success",
            "Attendance Recorded",
            (
                f"{new_log['employee_code']} {action_text} successful at "
                f"{new_log['log_time']}."
            ),
        )
    except ValueError as error:
        set_attendance_notice(
            "warning",
            "Attendance Blocked",
            str(error),
        )
    finally:
        reset_stability_state()


def render_attendance_status(run_camera, selected_camera_config):
    recognized_employee_code = st.session_state.get(
        f"{PAGE_KEY}_recognized_employee_code"
    )
    recognized_confidence = st.session_state.get(f"{PAGE_KEY}_recognized_confidence")
    recognized_threshold = st.session_state.get(f"{PAGE_KEY}_recognized_threshold")
    stable_count = st.session_state.get(f"{PAGE_KEY}_stable_frame_count", 0)
    display_stable_count = min(stable_count, AUTO_MATCH_REQUIRED_FRAMES)
    cooldown_remaining = 0

    if recognized_employee_code:
        cooldown_remaining = get_log_cooldown_remaining(
            recognized_employee_code,
            selected_camera_config,
        )

    with st.container(border=True):
        st.subheader("Recognition Result")

        if recognized_employee_code:
            code_col, confidence_col = st.columns([1.1, 1])
            with code_col:
                st.caption("Employee Code")
                st.markdown(f"### {recognized_employee_code}")
            with confidence_col:
                st.metric("Confidence", f"{recognized_confidence:.2f}")

            threshold_text = (
                "N/A"
                if recognized_threshold is None
                else f"{recognized_threshold:.2f}"
            )
            st.write(f"**Current threshold:** {threshold_text}")

            latest_log = get_latest_attendance_log(
                recognized_employee_code,
                log_type=selected_camera_config["log_type"],
            )
            if latest_log:
                st.caption(
                    f"Latest {get_action_text(selected_camera_config)} event for this employee"
                )
                log_col_1, log_col_2 = st.columns(2)
                log_col_1.write(f"**Type:** {latest_log['log_type']}")
                log_col_2.write(f"**Camera:** {latest_log['camera_source']}")
                st.write(f"**Time:** {latest_log['log_time']}")
            else:
                st.caption(
                    f"No previous {get_action_text(selected_camera_config)} record found for this employee."
                )
        else:
            st.info("No valid recognition result yet. Keep one face centered in the frame.")

    with st.container(border=True):
        st.subheader("Auto Attendance Mode")
        st.caption(selected_camera_config["description"])
        st.write(f"**Current mode:** {selected_camera_config['title']}")

        if not run_camera:
            st.info("Turn on 'Run Camera' to start touchless attendance.")
            return

        if not recognized_employee_code:
            st.info("Waiting for one valid face so the system can start stabilizing recognition.")
            st.progress(
                0.0,
                text=(
                    "Stable recognition progress: "
                    f"0 / {AUTO_MATCH_REQUIRED_FRAMES} frames"
                ),
            )
            return

        progress_value = min(stable_count / AUTO_MATCH_REQUIRED_FRAMES, 1.0)
        st.progress(
            progress_value,
            text=(
                "Stable recognition progress: "
                f"{display_stable_count} / {AUTO_MATCH_REQUIRED_FRAMES} frames"
            ),
        )

        if cooldown_remaining > 0:
            st.warning(
                f"Cooldown active for {recognized_employee_code}. Please wait {cooldown_remaining} second(s)."
            )
            return

        if stable_count < AUTO_MATCH_REQUIRED_FRAMES:
            remaining_frames = AUTO_MATCH_REQUIRED_FRAMES - stable_count
            remaining_seconds = remaining_frames * CAMERA_INTERVAL_SECONDS
            st.info(
                f"Hold still for about {remaining_seconds:.1f}s more "
                f"({remaining_frames} stable frame(s)) to trigger automatic attendance."
            )
            return

        st.success("Stable recognition confirmed. Recording attendance automatically...")


control_col_1, control_col_2, control_col_3 = st.columns([1.2, 0.8, 1.4], gap="large")
with control_col_1:
    selected_camera_index = st.selectbox(
        "Choose camera",
        [0, 1],
        format_func=lambda index: get_camera_config(index)["title"],
        key=f"{PAGE_KEY}_camera_index",
    )
with control_col_2:
    run_camera = st.checkbox("Run Camera", key=f"{PAGE_KEY}_run_camera")

selected_camera_config = get_camera_config(selected_camera_index)

with control_col_3:
    with st.container(border=True):
        st.caption("Auto Attendance Flow")
        st.markdown(f"**{selected_camera_config['title']}**")
        st.write(selected_camera_config["description"])
        st.caption(
            f"Requirement: {AUTO_MATCH_REQUIRED_FRAMES} stable frames before the system auto logs attendance."
        )

previous_camera_index = st.session_state.get(f"{PAGE_KEY}_active_camera_index")
if previous_camera_index != selected_camera_index:
    st.session_state[f"{PAGE_KEY}_active_camera_index"] = selected_camera_index
    release_camera(st.session_state, PAGE_KEY)
    clear_tracking_state()
    clear_attendance_notice()


@st.fragment(run_every=camera_run_every)
def render_dynamic_attendance():
    selected_camera_config = get_camera_config(st.session_state[f"{PAGE_KEY}_camera_index"])
    run_camera = st.session_state.get(f"{PAGE_KEY}_run_camera", False)

    render_attendance_notice()
    preview_col, details_col = st.columns([1.8, 0.95], gap="large")

    with preview_col:
        with st.container(border=True):
            st.subheader("Live Camera")
            st.caption(
                "Keep exactly one face inside the frame. When recognition stays stable long enough, attendance is recorded automatically."
            )

            if not run_camera:
                release_camera(st.session_state, PAGE_KEY)
                clear_tracking_state()
                st.info("Turn on 'Run Camera' to start attendance recognition.")
            else:
                cap = get_or_create_camera(
                    st.session_state,
                    PAGE_KEY,
                    st.session_state[f"{PAGE_KEY}_camera_index"],
                )

                if not cap.isOpened():
                    release_camera(st.session_state, PAGE_KEY)
                    clear_tracking_state()
                    st.error("Cannot open camera")
                else:
                    ret, frame = cap.read()
                    if not ret:
                        release_camera(st.session_state, PAGE_KEY)
                        clear_tracking_state()
                        st.error("Cannot read frame from camera")
                    else:
                        faces_list = update_detected_faces(
                            st.session_state,
                            PAGE_KEY,
                            frame,
                            detect_faces,
                        )

                        annotated_frame = frame.copy()
                        recognized_match = None
                        prediction = None
                        model_error = False

                        try:
                            prediction, _ = get_or_update_prediction(
                                st.session_state,
                                PAGE_KEY,
                                frame,
                                faces_list,
                                crop_and_resize_face,
                                predict_face,
                                camera_index=st.session_state[f"{PAGE_KEY}_camera_index"],
                                padding_ratio=RECOGNITION_PADDING_RATIO,
                            )
                        except Exception:
                            prediction = None
                            model_error = True

                        for (x, y, w, h) in faces_list:
                            if len(faces_list) == 1 and prediction is not None:
                                if prediction["is_match"]:
                                    recognized_match = prediction
                                    label_text = (
                                        f"{prediction['display_code']} "
                                        f"({prediction['confidence']:.2f})"
                                    )
                                    box_color = (0, 255, 0)
                                else:
                                    label_text = (
                                        f"Unknown ({prediction['confidence']:.2f} > "
                                        f"{prediction['match_threshold']:.2f})"
                                    )
                                    box_color = (0, 0, 255)
                            elif model_error:
                                label_text = "Model error"
                                box_color = (0, 0, 255)
                            else:
                                label_text = "Face detected"
                                box_color = (0, 215, 255)

                            cv2.rectangle(
                                annotated_frame,
                                (x, y),
                                (x + w, y + h),
                                box_color,
                                2,
                            )
                            cv2.putText(
                                annotated_frame,
                                label_text,
                                (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8,
                                box_color,
                                2,
                            )

                        if len(faces_list) == 1 and recognized_match is not None:
                            st.session_state[f"{PAGE_KEY}_recognized_employee_code"] = (
                                recognized_match["display_code"]
                            )
                            st.session_state[f"{PAGE_KEY}_recognized_confidence"] = (
                                recognized_match["confidence"]
                            )
                            st.session_state[f"{PAGE_KEY}_recognized_threshold"] = (
                                recognized_match["match_threshold"]
                            )
                            update_stability_state(recognized_match)
                            attempt_auto_attendance(selected_camera_config)
                        else:
                            clear_tracking_state()

                        cv2.putText(
                            annotated_frame,
                            f"Faces detected: {len(faces_list)}",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            2,
                        )

                        frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                        st.image(frame_rgb, channels="RGB")

    with details_col:
        render_attendance_status(run_camera, selected_camera_config)


render_dynamic_attendance()






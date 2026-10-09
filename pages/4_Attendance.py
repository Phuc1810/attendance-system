from datetime import datetime, timedelta
import math
import time
import cv2
import streamlit as st

from core.camera_stream import (
    STREAM_FRAME_INTERVAL,
    STREAM_SLEEP_INTERVAL,
    get_or_create_camera,
    get_or_update_prediction,
    release_camera,
    release_inactive_cameras,
    render_stream_frame,
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
RECOGNITION_PADDING_RATIO = 0.18
AUTO_MATCH_REQUIRED_FRAMES = 3
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

if f"{PAGE_KEY}_is_running" not in st.session_state:
    st.session_state[f"{PAGE_KEY}_is_running"] = False

release_inactive_cameras(st.session_state, PAGE_KEY)

st.title("📷 Face Attendance Camera (Native Live)")
st.markdown("Automatic Check-in/Check-out via high-performance Native OpenCV Camera.")

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
        clear_attendance_notice()
        return None
    return notice

def render_attendance_notice():
    notice = get_active_notice()
    if not notice:
        return
    icon = NOTICE_ICONS.get(notice["level"], NOTICE_ICONS["info"])
    st.toast(f"**{notice['title']}**: {notice['message']}", icon=icon)
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

def attempt_auto_attendance(camera_config, selected_camera_idx):
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
            f"A recent {action_text} for {recognized_code} was recorded. Wait {cooldown_remaining}s.",
        )
        reset_stability_state()
        return

    try:
        new_log = create_attendance_from_camera(
            employee_code=recognized_code,
            confidence=recognized_confidence,
            camera_index=selected_camera_idx,
        )
        st.session_state[f"{PAGE_KEY}_latest_success_log"] = new_log
        set_attendance_notice(
            "success",
            "Attendance Logged",
            f"{new_log['employee_code']} {action_text} successful at {new_log['log_time']}.",
        )
    except ValueError as error:
        set_attendance_notice(
            "warning",
            "Attendance Blocked",
            str(error),
        )
    finally:
        reset_stability_state()


# --- HEADER CONTROLS ---
control_col_1, control_col_2, control_col_3 = st.columns([1.2, 1.2, 1.4], gap="large")
with control_col_1:
    selected_camera_index = st.selectbox(
        "Choose camera",
        [0, 1],
        format_func=lambda index: get_camera_config(index)["title"],
        key=f"{PAGE_KEY}_camera_index",
    )

selected_camera_config = get_camera_config(selected_camera_index)

with control_col_2:
    st.caption("Camera Controls")
    btn_start_col, btn_stop_col = st.columns(2)
    with btn_start_col:
        if st.button("▶️ Bật Live", type="primary", use_container_width=True):
            st.session_state[f"{PAGE_KEY}_is_running"] = True
    with btn_stop_col:
        if st.button("⏹️ Tắt", use_container_width=True):
            st.session_state[f"{PAGE_KEY}_is_running"] = False
            release_camera(st.session_state, PAGE_KEY)
            clear_tracking_state()
            st.rerun()

with control_col_3:
    with st.container(border=True):
        st.caption("Active Mode")
        st.markdown(f"**{selected_camera_config['title']}**")
        st.caption(f"{selected_camera_config['description']} (Requires {AUTO_MATCH_REQUIRED_FRAMES} stable frames)")

previous_camera_index = st.session_state.get(f"{PAGE_KEY}_active_camera_index")
if previous_camera_index != selected_camera_index:
    st.session_state[f"{PAGE_KEY}_active_camera_index"] = selected_camera_index
    release_camera(st.session_state, PAGE_KEY)
    clear_tracking_state()
    clear_attendance_notice()

st.divider()

render_attendance_notice()
preview_col, details_col = st.columns([1.5, 1], gap="large")

with preview_col:
    with st.container(border=True):
        st.subheader("Live Preview (Native 30 FPS)")
        st.caption("Khung hình đọc trực tiếp từ DirectShow webcam với độ trễ 0ms.")
        video_placeholder = st.empty()

with details_col:
    status_card_placeholder = st.empty()
    latest_log_placeholder = st.empty()

is_running = st.session_state.get(f"{PAGE_KEY}_is_running", False)

if not is_running:
    video_placeholder.info("🟢 Camera đang tắt. Bấm nút **'▶️ Bật Live'** ở trên để phát video trực tiếp.")
    with status_card_placeholder.container(border=True):
        st.subheader("Recognition Status")
        st.info("Bật camera để bắt đầu nhận diện và điểm danh tự động.")
else:
    cap = get_or_create_camera(st.session_state, PAGE_KEY, selected_camera_index)
    if cap is None or not cap.isOpened():
        st.session_state[f"{PAGE_KEY}_is_running"] = False
        release_camera(st.session_state, PAGE_KEY)
        clear_tracking_state()
        video_placeholder.error("Không thể mở thiết bị Camera. Vui lòng kiểm tra webcam.")
    else:
        frame_counter = 0
        fps_start = time.time()
        last_video_time = 0.0
        last_ui_time = 0.0
        last_rendered_faces = -1
        last_rendered_code = None
        last_rendered_pred_code = None
        last_rendered_pred_match = None
        last_rendered_cnt = -1

        # Vòng lặp phát video trực tiếp mượt mà liên tục (Live Loop)
        while st.session_state.get(f"{PAGE_KEY}_is_running", False):
            ret, frame = cap.read()
            if not ret or frame is None:
                video_placeholder.warning("Mất tín hiệu camera hoặc đang khởi tạo...")
                time.sleep(0.03)
                continue

            frame_counter += 1
            faces_list = update_detected_faces(st.session_state, PAGE_KEY, frame, detect_faces)

            annotated_frame = frame.copy()
            recognized_match = None
            prediction = None
            model_error = False

            try:
                prediction, is_new_prediction = get_or_update_prediction(
                    st.session_state,
                    PAGE_KEY,
                    frame,
                    faces_list,
                    crop_and_resize_face,
                    predict_face,
                    camera_index=selected_camera_index,
                    padding_ratio=RECOGNITION_PADDING_RATIO,
                )
            except Exception:
                prediction = None
                is_new_prediction = False
                model_error = True

            # Vẽ Bounding Box trực quan theo trạng thái nhận diện thực tế
            for (x, y, w, h) in faces_list:
                if len(faces_list) > 1:
                    label_text = f"Multiple faces ({len(faces_list)})"
                    box_color = (0, 165, 255)  # Màu cam cảnh báo
                elif model_error:
                    label_text = "Model error"
                    box_color = (0, 0, 255)    # Màu đỏ lỗi
                elif prediction is not None:
                    if prediction.get("is_match", False):
                        recognized_match = prediction
                        label_text = f"{prediction['display_code']} ({prediction['confidence']:.2f})"
                        box_color = (0, 255, 0)  # Màu xanh lá nhận diện chuẩn xác
                    else:
                        label_text = f"Unknown ({prediction['confidence']:.2f})"
                        box_color = (0, 0, 255)  # Màu đỏ chưa đăng ký
                else:
                    # Đang quét / phân tích khuôn mặt lần đầu (chưa có kết quả)
                    label_text = "Scanning..."
                    box_color = (0, 215, 255)  # Màu vàng cam đang quét

                cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), box_color, 2)
                cv2.putText(
                    annotated_frame,
                    label_text,
                    (x, max(20, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    box_color,
                    2,
                )

            # Đếm FPS thời gian thực
            elapsed = time.time() - fps_start
            fps = frame_counter / elapsed if elapsed > 0 else 0
            cv2.putText(
                annotated_frame,
                f"LIVE: {fps:.1f} FPS | Faces: {len(faces_list)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )

            # Cập nhật kết quả nhận diện & điểm danh tự động (chỉ tăng stable count khi có prediction mới ngầm)
            if len(faces_list) == 1 and recognized_match is not None:
                st.session_state[f"{PAGE_KEY}_recognized_employee_code"] = recognized_match["display_code"]
                st.session_state[f"{PAGE_KEY}_recognized_confidence"] = recognized_match["confidence"]
                st.session_state[f"{PAGE_KEY}_recognized_threshold"] = recognized_match["match_threshold"]
                if is_new_prediction:
                    update_stability_state(recognized_match)
                attempt_auto_attendance(selected_camera_config, selected_camera_index)
            else:
                clear_tracking_state()

            # Điều tiết hiển thị video Zero-Delay (Golden FPS ~14.7 FPS, Turbo-JPEG nhẹ ~20KB)
            now = time.time()
            if now - last_video_time >= STREAM_FRAME_INTERVAL:
                render_stream_frame(video_placeholder, annotated_frame)
                last_video_time = now

            # Điều tiết cập nhật widget trạng thái: chỉ vẽ lại khi có kết quả mới, đổi số mặt, hoặc định kỳ 0.25s
            rec_code = st.session_state.get(f"{PAGE_KEY}_recognized_employee_code")
            stable_cnt = st.session_state.get(f"{PAGE_KEY}_stable_frame_count", 0)
            pred_code = prediction.get("display_code") if prediction else None
            pred_match = prediction.get("is_match") if prediction else None

            ui_needed = (
                is_new_prediction
                or len(faces_list) != last_rendered_faces
                or rec_code != last_rendered_code
                or pred_code != last_rendered_pred_code
                or pred_match != last_rendered_pred_match
                or stable_cnt != last_rendered_cnt
                or (now - last_ui_time >= 0.25)
            )

            if ui_needed:
                last_ui_time = now
                last_rendered_faces = len(faces_list)
                last_rendered_code = rec_code
                last_rendered_pred_code = pred_code
                last_rendered_pred_match = pred_match
                last_rendered_cnt = stable_cnt

                with status_card_placeholder.container(border=True):
                    st.subheader("Recognition Status")
                    if len(faces_list) == 0:
                        st.markdown(
                            """
                            <div style="text-align: center; padding: 20px;">
                                <h3 style="color: #64748B;">Looking for face...</h3>
                                <p style="color: #94A3B8;">Please step into the frame</p>
                            </div>
                            """, unsafe_allow_html=True
                        )
                    elif len(faces_list) > 1:
                        st.markdown(
                            f"""
                            <div style="padding: 15px; border-radius: 8px; background-color: #FEF3C7; border: 1px solid #F59E0B; text-align: center;">
                                <h4 style="margin: 0; color: #B45309;">⚠️ Multiple Faces Detected</h4>
                                <p style="margin: 5px 0 0 0; color: #92400E;">Phát hiện {len(faces_list)} khuôn mặt. Vui lòng chỉ đứng 1 người trước camera.</p>
                            </div>
                            """, unsafe_allow_html=True
                        )
                    elif model_error:
                        st.error("❌ Lỗi mô hình: Không thể thực hiện nhận diện.")
                    elif prediction is None:
                        st.markdown(
                            """
                            <div style="padding: 15px; border-radius: 8px; background-color: #EFF6FF; border: 1px solid #3B82F6; text-align: center;">
                                <h4 style="margin: 0; color: #1D4ED8;">🔍 Analyzing Face...</h4>
                                <p style="margin: 5px 0 0 0; color: #1E40AF;">Đang phân tích khuôn mặt, vui lòng giữ yên...</p>
                            </div>
                            """, unsafe_allow_html=True
                        )
                    elif not prediction.get("is_match", False):
                        st.markdown(
                            f"""
                            <div style="padding: 15px; border-radius: 8px; background-color: #FEE2E2; border: 1px solid #EF4444; text-align: center;">
                                <h4 style="margin: 0; color: #B91C1C;">❌ Unknown Employee</h4>
                                <p style="margin: 5px 0 0 0; color: #991B1B;">Khuôn mặt chưa được đăng ký trong hệ thống.</p>
                                <p style="margin: 4px 0 0 0; font-size: 0.85em; color: #7F1D1D;">Độ tin cậy: {prediction['confidence']:.2f} > Ngưỡng: {prediction['match_threshold']:.2f}</p>
                            </div>
                            """, unsafe_allow_html=True
                        )
                    else:
                        # Khớp nhân viên hợp lệ (Target Locked)
                        disp_cnt = min(stable_cnt, AUTO_MATCH_REQUIRED_FRAMES)
                        st.markdown(f"**🎯 Target Locked: {rec_code}**")
                        st.progress(
                            min(stable_cnt / AUTO_MATCH_REQUIRED_FRAMES, 1.0),
                            text=f"Stabilizing... ({disp_cnt} / {AUTO_MATCH_REQUIRED_FRAMES} frames)",
                        )
                        if stable_cnt >= AUTO_MATCH_REQUIRED_FRAMES:
                            st.success(f"✅ Xác thực thành công: **{rec_code}**! Đang ghi nhận điểm danh...")
                        else:
                            st.info(f"Giữ yên mặt thêm {AUTO_MATCH_REQUIRED_FRAMES - stable_cnt} frame...")

                # Hiển thị thẻ log điểm danh mới nhất hoặc thông tin nhân viên vừa quét
                latest_log = st.session_state.get(f"{PAGE_KEY}_latest_success_log")
                if rec_code:
                    rec_conf = st.session_state.get(f"{PAGE_KEY}_recognized_confidence", 0)
                    latest_log_placeholder.markdown(
                        f"""
                        <div style="background-color: #FFFFFF; padding: 15px; border-radius: 8px; border: 1px solid #10B981; margin-top: 15px;">
                            <h4 style="margin:0; color: #10B981;">{rec_code}</h4>
                            <p style="margin: 5px 0 0 0; color: #334155;">Confidence: <b>{rec_conf:.2f}</b></p>
                        </div>
                        """, unsafe_allow_html=True
                    )
                elif latest_log:
                    latest_log_placeholder.markdown(
                        f"""
                        <div style="background-color: #F0FDF4; padding: 15px; border-radius: 8px; border: 1px solid #10B981; margin-top: 15px;">
                            <span style="background-color: #10B981; color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.8em; font-weight: bold;">{latest_log.get('log_type', 'IN')}</span>
                            <h4 style="margin:5px 0 0 0; color: #047857;">{latest_log.get('employee_code')}</h4>
                            <p style="margin: 3px 0 0 0; font-size: 0.9em; color: #334155;">Thời gian: <b>{latest_log.get('log_time')}</b></p>
                        </div>
                        """, unsafe_allow_html=True
                    )
                else:
                    latest_log_placeholder.empty()

            time.sleep(STREAM_SLEEP_INTERVAL)

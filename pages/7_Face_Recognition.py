import time
import cv2
import streamlit as st

from core.camera_stream import (
    get_or_create_camera,
    get_or_update_prediction,
    release_camera,
    release_inactive_cameras,
    update_detected_faces,
)
from core.face_detect import detect_faces
from core.face_recognizer import predict_face, train_model
from core.save_face import crop_and_resize_face
from db.database import initialize_database

PAGE_KEY = "face_recognition"
RECOGNITION_PADDING_RATIO = 0.18

initialize_database()

if f"{PAGE_KEY}_is_running" not in st.session_state:
    st.session_state[f"{PAGE_KEY}_is_running"] = False

release_inactive_cameras(st.session_state, PAGE_KEY)

st.title("🎯 Face Recognition (Native Live)")
st.caption("Train the recognition model, then test how well it identifies faces in realtime via Native OpenCV Camera.")

# --- STEP 1: TRAIN MODEL ---
train_col, summary_col = st.columns([1.1, 1], gap="large")

with train_col:
    with st.container(border=True):
        st.subheader("Step 1: Train Model")
        st.caption("Retrain after adding or deleting employees or face images so the model stays up to date.")

        if st.button("Train Recognition Model", type="primary", use_container_width=True):
            try:
                result = train_model()
                st.session_state[f"{PAGE_KEY}_train_result"] = result
                st.session_state[f"{PAGE_KEY}_train_error"] = None
                st.toast("✅ Huấn luyện mô hình nhận diện thành công!")
            except Exception as error:
                st.session_state[f"{PAGE_KEY}_train_result"] = None
                st.session_state[f"{PAGE_KEY}_train_error"] = str(error)
                st.toast(f"⚠️ Huấn luyện thất bại: {error}")

with summary_col:
    with st.container(border=True):
        st.subheader("Training Summary")
        train_error = st.session_state.get(f"{PAGE_KEY}_train_error")
        train_result = st.session_state.get(f"{PAGE_KEY}_train_result")

        if train_error:
            st.error(f"Training failed: {train_error}")
        elif train_result:
            metric_col_1, metric_col_2 = st.columns(2)
            metric_col_1.metric("Images", train_result["num_images"])
            metric_col_2.metric("People", train_result["num_people"])

            with st.expander("Employee labels used in the model"):
                st.json(train_result["label_to_code"])

            with st.expander("Unknown rejection thresholds"):
                st.json(train_result["code_thresholds"])

            with st.expander("Camera-aware threshold profiles"):
                st.json(train_result["camera_profiles"])
        else:
            st.info("Train the model once to view the summary and current thresholds.")

st.divider()

# --- STEP 2: TEST RECOGNITION (NATIVE LIVE LOOP) ---
st.subheader("Step 2: Test Recognition")

control_col_1, control_col_2, control_col_3 = st.columns([1, 1.2, 1.2], gap="large")
with control_col_1:
    selected_camera_index = st.selectbox(
        "Choose camera",
        [0, 1],
        format_func=lambda idx: f"Camera {idx}",
        key=f"{PAGE_KEY}_camera_index",
    )
with control_col_2:
    st.caption("Camera Controls")
    btn_start_col, btn_stop_col = st.columns(2)
    with btn_start_col:
        if st.button("▶️ Bật Live Test", type="primary", use_container_width=True):
            st.session_state[f"{PAGE_KEY}_is_running"] = True
    with btn_stop_col:
        if st.button("⏹️ Tắt", use_container_width=True):
            st.session_state[f"{PAGE_KEY}_is_running"] = False
            release_camera(st.session_state, PAGE_KEY)
            st.rerun()

with control_col_3:
    with st.container(border=True):
        st.caption("Test Goal")
        st.write("Xác minh mô hình nhận diện trực tiếp trên luồng Native Camera thời gian thực.")

previous_camera_index = st.session_state.get(f"{PAGE_KEY}_active_camera_index")
if previous_camera_index != selected_camera_index:
    st.session_state[f"{PAGE_KEY}_active_camera_index"] = selected_camera_index
    release_camera(st.session_state, PAGE_KEY)

preview_col, status_col = st.columns([1.5, 1], gap="large")

with preview_col:
    with st.container(border=True):
        st.subheader("Live Recognition (Native 30 FPS)")
        video_placeholder = st.empty()

with status_col:
    status_card_placeholder = st.empty()

is_running = st.session_state.get(f"{PAGE_KEY}_is_running", False)

if not is_running:
    video_placeholder.info("Camera đang tắt. Bấm **'▶️ Bật Live Test'** để kiểm tra nhận diện trực tiếp.")
    with status_card_placeholder.container(border=True):
        st.subheader("Recognition Status")
        st.write("**Status:** Stopped")
        st.info("Bật camera để xem kết quả nhận diện thời gian thực.")
else:
    cap = get_or_create_camera(st.session_state, PAGE_KEY, selected_camera_index)
    if cap is None or not cap.isOpened():
        st.session_state[f"{PAGE_KEY}_is_running"] = False
        release_camera(st.session_state, PAGE_KEY)
        video_placeholder.error("Không thể mở thiết bị Camera.")
    else:
        frame_counter = 0
        fps_start = time.time()
        last_video_time = 0.0
        last_ui_time = 0.0
        last_rendered_faces = -1
        last_rendered_pred_id = None

        while st.session_state.get(f"{PAGE_KEY}_is_running", False):
            ret, frame = cap.read()
            if not ret or frame is None:
                video_placeholder.warning("Đang chờ khung hình...")
                time.sleep(0.03)
                continue

            frame_counter += 1
            faces_list = update_detected_faces(st.session_state, PAGE_KEY, frame, detect_faces)
            annotated_frame = frame.copy()

            prediction = None
            is_new_prediction = False
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

            for (x, y, w, h) in faces_list:
                if len(faces_list) == 1 and prediction is not None:
                    if prediction.get("is_match", False):
                        label_text = f"{prediction['display_code']} ({prediction['confidence']:.2f})"
                        box_color = (0, 255, 0)
                    else:
                        label_text = f"Unknown ({prediction['confidence']:.2f} > {prediction['match_threshold']:.2f})"
                        box_color = (0, 0, 255)
                elif model_error:
                    label_text = "Model error"
                    box_color = (0, 0, 255)
                else:
                    label_text = "Face detected"
                    box_color = (0, 215, 255)

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

            # Điều tiết hiển thị video để không làm tràn hàng đợi WebSocket (tránh tích tụ trễ 1-2s)
            now = time.time()
            if now - last_video_time >= 0.035:  # Giới hạn ~25-28 FPS mượt mà
                frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                last_video_time = now

            # Điều tiết cập nhật widget trạng thái: chỉ vẽ lại khi có kết quả mới, đổi số mặt, hoặc định kỳ 0.25s
            pred_id = (prediction.get("display_code"), prediction.get("is_match")) if prediction else None
            ui_needed = (
                is_new_prediction
                or len(faces_list) != last_rendered_faces
                or pred_id != last_rendered_pred_id
                or (now - last_ui_time >= 0.25)
            )

            if ui_needed:
                last_ui_time = now
                last_rendered_faces = len(faces_list)
                last_rendered_pred_id = pred_id

                with status_card_placeholder.container(border=True):
                    st.subheader("Recognition Status")
                    metric_col_1, metric_col_2 = st.columns(2)
                    metric_col_1.metric("Camera", selected_camera_index)
                    metric_col_2.metric("Faces Detected", len(faces_list))

                    if len(faces_list) == 0:
                        st.write("**Status:** Đang tìm khuôn mặt...")
                        st.caption("Hãy đứng thẳng trước camera.")
                    elif len(faces_list) > 1:
                        st.write("**Status:** Phát hiện nhiều khuôn mặt")
                        st.warning("Vui lòng chỉ để 1 người trong khung hình.")
                    else:
                        st.write("**Status:** Đang nhận diện...")
                        if prediction:
                            if prediction.get("is_match", False):
                                st.success(f"Khớp nhân viên: **{prediction['display_code']}**")
                                st.write(f"**Confidence:** {prediction['confidence']:.2f}")
                                st.write(f"**Threshold:** {prediction['match_threshold']:.2f}")
                            else:
                                st.warning("Kết quả: **Unknown (Chưa nhận diện)**")
                                st.write(f"**Confidence:** {prediction['confidence']:.2f}")
                                st.write(f"**Threshold:** {prediction['match_threshold']:.2f}")
                        elif model_error:
                            st.error("Lỗi khi chạy mô hình nhận diện.")

            time.sleep(0.005)

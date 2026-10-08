from pathlib import Path
import cv2
import numpy as np
import streamlit as st

from core.camera_stream import (
    get_or_create_camera,
    release_camera,
    release_inactive_cameras,
    update_detected_faces,
    annotate_faces,
)
from core.face_detect import detect_faces
from core.save_face import crop_and_resize_face, save_face_image
from db.database import get_all_employees, initialize_database

PAGE_KEY = "register_face"
CAMERA_INTERVAL_SECONDS = 0.1
FACE_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "faces"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

initialize_database()

if f"{PAGE_KEY}_run_camera" not in st.session_state:
    st.session_state[f"{PAGE_KEY}_run_camera"] = False

release_inactive_cameras(st.session_state, PAGE_KEY)

camera_run_every = (
    CAMERA_INTERVAL_SECONDS
    if st.session_state.get(f"{PAGE_KEY}_run_camera", False)
    else None
)

def count_employee_images(employee_code):
    employee_folder = FACE_DATA_DIR / employee_code
    if not employee_folder.exists():
        return 0
    return sum(
        1 for file_path in employee_folder.iterdir()
        if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS
    )

st.title("🧑‍💻 Register Face (Native)")
st.markdown("Collect face images for each employee to train the recognition model.")

employees = get_all_employees()

if not employees:
    st.warning("⚠️ No employees found. Please add employees first in the Management tab.")
    st.stop()

employee_map = {
    employee_code: {
        "employee_code": employee_code,
        "name": name,
        "department": department,
    }
    for _, employee_code, name, department in employees
}

selector_col, summary_col = st.columns([1, 1], gap="large")

with selector_col:
    selected_employee_code = st.selectbox(
        "Select Employee",
        list(employee_map.keys()),
        format_func=lambda code: f"{code} - {employee_map[code]['name']} ({employee_map[code]['department']})",
    )

selected_employee = employee_map[selected_employee_code]
saved_image_count = count_employee_images(selected_employee_code)

with summary_col:
    st.markdown(
        f"""
        <div style="background-color: #FFFFFF; padding: 15px; border-radius: 8px; border: 1px solid #E2E8F0; box-shadow: 0 1px 3px rgba(0,0,0,0.1)">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <h3 style="margin:0; color: #1E3A8A;">{selected_employee['employee_code']}</h3>
                <span style="background-color: #F1F5F9; color: #475569; padding: 4px 10px; border-radius: 20px; font-weight: 500; font-size: 14px;">{saved_image_count} images</span>
            </div>
            <h5 style="margin:5px 0 0 0; color: #64748B;">{selected_employee['name']}</h5>
            <p style="margin: 5px 0 0 0; color: #94A3B8;">{selected_employee['department']}</p>
        </div>
        """, unsafe_allow_html=True
    )

st.divider()

tab_camera, tab_upload = st.tabs(["📷 Camera Capture (Native)", "📁 Upload Images"])

# --- TAB CAMERA CAPTURE (NATIVE) ---
with tab_camera:
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1, 1, 1.2])
    with ctrl_col1:
        cam_idx = st.selectbox("Choose Camera", [0, 1], key=f"{PAGE_KEY}_camera_index")
    with ctrl_col2:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        run_camera = st.toggle("🎥 Bật Camera Preview", key=f"{PAGE_KEY}_run_camera")

    previous_camera_index = st.session_state.get(f"{PAGE_KEY}_active_camera_index")
    if previous_camera_index != cam_idx:
        st.session_state[f"{PAGE_KEY}_active_camera_index"] = cam_idx
        release_camera(st.session_state, PAGE_KEY)

    @st.fragment(run_every=camera_run_every)
    def render_register_camera():
        run_cam = st.session_state.get(f"{PAGE_KEY}_run_camera", False)
        cam_index = st.session_state.get(f"{PAGE_KEY}_camera_index", 0)

        cam_col, info_col = st.columns([1.5, 1], gap="large")

        with cam_col:
            with st.container(border=True):
                st.subheader("Live Preview (Native)")

                if not run_cam:
                    release_camera(st.session_state, PAGE_KEY)
                    st.info("Camera đang tắt. Bật 'Bật Camera Preview' để căn chỉnh khuôn mặt.")
                else:
                    cap = get_or_create_camera(st.session_state, PAGE_KEY, cam_index)
                    if cap is None or not cap.isOpened():
                        release_camera(st.session_state, PAGE_KEY)
                        st.error("Không thể mở thiết bị Camera.")
                    else:
                        ret, frame = cap.read()
                        if not ret or frame is None:
                            release_camera(st.session_state, PAGE_KEY)
                            st.error("Không thể đọc khung hình từ camera.")
                        else:
                            # Lưu frame gốc sạch vào session_state để dùng khi bấm nút Chụp
                            st.session_state[f"{PAGE_KEY}_latest_clean_frame"] = frame.copy()

                            faces = update_detected_faces(st.session_state, PAGE_KEY, frame, detect_faces)
                            annotated = annotate_faces(frame, faces)

                            frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                            st.image(frame_rgb, channels="RGB", use_container_width=True)

        with info_col:
            with st.container(border=True):
                st.subheader("Capture Controls")
                st.caption("1. Bật camera và nhìn thẳng vào ống kính.")
                st.caption("2. Đảm bảo chỉ có 1 khuôn mặt trong khung hình.")
                st.caption("3. Bấm nút Chụp để lưu mẫu nhận diện.")

                st.markdown("<br>", unsafe_allow_html=True)
                capture_pressed = st.button("📸 Chụp & Lưu khuôn mặt", type="primary", use_container_width=True)

                if capture_pressed:
                    clean_frame = st.session_state.get(f"{PAGE_KEY}_latest_clean_frame")
                    if not run_cam or clean_frame is None:
                        st.toast("⚠️ Vui lòng bật camera preview trước khi chụp!")
                    else:
                        detected_faces, _ = detect_faces(clean_frame)
                        if len(detected_faces) == 0:
                            st.toast("⚠️ Không tìm thấy khuôn mặt trong ảnh! Vui lòng thử lại.")
                        elif len(detected_faces) > 1:
                            st.toast("⚠️ Phát hiện nhiều khuôn mặt! Hãy đảm bảo chỉ có 1 người trong khung hình.")
                        else:
                            x, y, w, h = detected_faces[0]
                            face_crop = crop_and_resize_face(clean_frame, (x, y, w, h))
                            save_path = save_face_image(selected_employee_code, face_crop)
                            st.toast(f"✅ Đã lưu ảnh khuôn mặt cho nhân viên {selected_employee_code}!")
                            st.rerun()

    render_register_camera()

# --- TAB UPLOAD IMAGES ---
with tab_upload:
    with st.container(border=True):
        st.subheader("Upload Existing Photos")
        st.caption("Use clear frontal face images. Max 1 person per image.")
        uploaded_files = st.file_uploader(
            "Upload one or more images",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
        )
        if uploaded_files and st.button("Process Uploaded Images", type="primary", use_container_width=True):
            stats = {"saved": 0, "no_face": 0, "multiple_faces": 0, "invalid": 0}
            for uploaded_file in uploaded_files:
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                if image is None:
                    stats["invalid"] += 1
                    continue
                faces, _ = detect_faces(image)
                if len(faces) == 0:
                    stats["no_face"] += 1
                    continue
                if len(faces) > 1:
                    stats["multiple_faces"] += 1
                    continue

                x, y, w, h = faces[0]
                face_crop = crop_and_resize_face(image, (x, y, w, h))
                save_face_image(selected_employee_code, face_crop)
                stats["saved"] += 1

            if stats["saved"]:
                st.toast(f"✅ Successfully saved {stats['saved']} image(s).")
            if stats["no_face"] or stats["multiple_faces"] or stats["invalid"]:
                st.toast(f"⚠️ Skipped {stats['no_face'] + stats['multiple_faces'] + stats['invalid']} invalid image(s).")
            st.rerun()

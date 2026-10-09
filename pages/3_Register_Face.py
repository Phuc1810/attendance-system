from pathlib import Path
import time
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
FACE_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "faces"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

initialize_database()

if f"{PAGE_KEY}_is_running" not in st.session_state:
    st.session_state[f"{PAGE_KEY}_is_running"] = False

release_inactive_cameras(st.session_state, PAGE_KEY)

def count_employee_images(employee_code):
    employee_folder = FACE_DATA_DIR / employee_code
    if not employee_folder.exists():
        return 0
    return sum(
        1 for file_path in employee_folder.iterdir()
        if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS
    )

st.title("🧑‍💻 Register Face (Native Live)")
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

tab_camera, tab_upload = st.tabs(["📷 Camera Capture (Native Live)", "📁 Upload Images"])

# --- TAB CAMERA CAPTURE (NATIVE LIVE) ---
with tab_camera:
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1, 1.2, 1.2])
    with ctrl_col1:
        cam_idx = st.selectbox("Choose Camera", [0, 1], key=f"{PAGE_KEY}_camera_index")
    with ctrl_col2:
        st.caption("Camera Controls")
        b_start_col, b_stop_col = st.columns(2)
        with b_start_col:
            if st.button("▶️ Bật Preview", type="primary", use_container_width=True):
                st.session_state[f"{PAGE_KEY}_is_running"] = True
        with b_stop_col:
            if st.button("⏹️ Tắt", use_container_width=True):
                st.session_state[f"{PAGE_KEY}_is_running"] = False
                release_camera(st.session_state, PAGE_KEY)
                st.rerun()

    previous_camera_index = st.session_state.get(f"{PAGE_KEY}_active_camera_index")
    if previous_camera_index != cam_idx:
        st.session_state[f"{PAGE_KEY}_active_camera_index"] = cam_idx
        release_camera(st.session_state, PAGE_KEY)

    cam_col, info_col = st.columns([1.5, 1], gap="large")

    with cam_col:
        with st.container(border=True):
            st.subheader("Live Preview (Native 30 FPS)")
            video_placeholder = st.empty()

    with info_col:
        with st.container(border=True):
            st.subheader("Capture Controls")
            st.caption("1. Bật camera preview và nhìn thẳng.")
            st.caption("2. Đảm bảo có đúng 1 khuôn mặt trong khung hình.")
            st.caption("3. Bấm nút Chụp để lưu ảnh mẫu nhận diện.")

            capture_btn_placeholder = st.empty()
            capture_status_placeholder = st.empty()

    is_running = st.session_state.get(f"{PAGE_KEY}_is_running", False)

    if not is_running:
        video_placeholder.info("Camera đang tắt. Bấm **'▶️ Bật Preview'** để căn chỉnh khuôn mặt.")
        capture_status_placeholder.info("Bật camera để kích hoạt nút chụp ảnh.")
    else:
        cap = get_or_create_camera(st.session_state, PAGE_KEY, cam_idx)
        if cap is None or not cap.isOpened():
            st.session_state[f"{PAGE_KEY}_is_running"] = False
            release_camera(st.session_state, PAGE_KEY)
            video_placeholder.error("Không thể mở thiết bị Camera.")
        else:
            frame_counter = 0
            fps_start = time.time()

            # Hiển thị nút chụp ảnh
            do_capture = capture_btn_placeholder.button("📸 Chụp & Lưu khuôn mặt", type="primary", use_container_width=True)

            while st.session_state.get(f"{PAGE_KEY}_is_running", False):
                ret, frame = cap.read()
                if not ret or frame is None:
                    video_placeholder.warning("Đang chờ khung hình...")
                    time.sleep(0.05)
                    continue

                frame_counter += 1
                faces = update_detected_faces(st.session_state, PAGE_KEY, frame, detect_faces)
                annotated = annotate_faces(frame, faces)

                elapsed = time.time() - fps_start
                fps = frame_counter / elapsed if elapsed > 0 else 0
                cv2.putText(
                    annotated,
                    f"LIVE: {fps:.1f} FPS",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2,
                )

                frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

                if len(faces) == 0:
                    capture_status_placeholder.warning("⚠️ Không thấy mặt: Hãy đứng vào giữa khung hình.")
                elif len(faces) > 1:
                    capture_status_placeholder.warning("⚠️ Nhiều mặt: Chỉ để 1 người trong khung hình.")
                else:
                    capture_status_placeholder.success("✅ Mặt sẵn sàng: Bấm 'Chụp & Lưu' để lưu mẫu.")

                # Nếu người dùng bấm Chụp
                if do_capture:
                    if len(faces) == 1:
                        face_crop = crop_and_resize_face(frame, faces[0])
                        save_path = save_face_image(selected_employee_code, face_crop)
                        st.toast(f"✅ Đã lưu ảnh thành công cho {selected_employee_code}!")
                        time.sleep(0.5)
                        st.rerun()
                    else:
                        st.toast("⚠️ Vui lòng đảm bảo có đúng 1 khuôn mặt trước khi bấm chụp!")
                        time.sleep(0.5)
                        st.rerun()

                time.sleep(0.01)

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

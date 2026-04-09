from pathlib import Path

import cv2
import numpy as np
import streamlit as st
st.set_page_config(page_title="Attendance Dashboard", layout="wide")

from core.camera_stream import (
    annotate_faces,
    get_or_create_camera,
    release_camera,
    release_inactive_cameras,
    update_detected_faces,
)
from core.face_detect import detect_faces
from core.save_face import crop_and_resize_face, save_face_image
from db.database import get_all_employees, initialize_database


PAGE_KEY = "register_face"
CAMERA_INTERVAL_SECONDS = 0.15
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
        1
        for file_path in employee_folder.iterdir()
        if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS
    )


@st.fragment(run_every=camera_run_every)
def render_live_camera():
    if not st.session_state.get(f"{PAGE_KEY}_run_camera", False):
        return

    cap = get_or_create_camera(
        st.session_state,
        PAGE_KEY,
        st.session_state[f"{PAGE_KEY}_camera_index"],
    )

    if not cap.isOpened():
        st.error("Cannot open camera")
        release_camera(st.session_state, PAGE_KEY)
        return

    ret, frame = cap.read()
    if not ret:
        st.error("Cannot read frame from camera")
        release_camera(st.session_state, PAGE_KEY)
        return

    faces_list = update_detected_faces(
        st.session_state,
        PAGE_KEY,
        frame,
        detect_faces,
    )

    annotated_frame = annotate_faces(frame, faces_list)
    frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
    st.image(frame_rgb, channels="RGB")


st.title("Register Face")
st.caption("Collect clean face images for each employee before training the recognition model.")

employees = get_all_employees()

if not employees:
    st.warning("No employees found. Please add employees first.")
    st.stop()

employee_map = {
    employee_code: {
        "employee_code": employee_code,
        "name": name,
        "department": department,
    }
    for _, employee_code, name, department in employees
}

selector_col, summary_col = st.columns([1.2, 1], gap="large")

with selector_col:
    selected_employee_code = st.selectbox(
        "Choose employee",
        list(employee_map.keys()),
        format_func=lambda code: (
            f"{code} - {employee_map[code]['name']} ({employee_map[code]['department']})"
        ),
    )

selected_employee = employee_map[selected_employee_code]
saved_image_count = count_employee_images(selected_employee_code)

with summary_col:
    with st.container(border=True):
        st.subheader("Selected Employee")
        summary_col_1, summary_col_2 = st.columns(2)
        summary_col_3, summary_col_4 = st.columns(2)

        with summary_col_1:
            st.caption("Employee Code")
            st.write(selected_employee["employee_code"])
        with summary_col_2:
            st.caption("Name")
            st.write(selected_employee["name"])
        with summary_col_3:
            st.caption("Department")
            st.write(selected_employee["department"])
        with summary_col_4:
            st.metric("Saved Images", saved_image_count)

tab_upload, tab_camera = st.tabs(["Upload Images", "Camera Capture"])

with tab_upload:
    with st.container(border=True):
        st.subheader("Upload Face Images")
        st.caption(
            "Use one face per image and include multiple head angles for better recognition quality."
        )

        uploaded_files = st.file_uploader(
            "Upload one or more images",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
        )

        if uploaded_files and st.button(
            "Process Uploaded Images",
            type="primary",
            use_container_width=True,
        ):
            stats = {
                "saved": 0,
                "no_face": 0,
                "multiple_faces": 0,
                "invalid": 0,
            }
            processing_details = []

            for uploaded_file in uploaded_files:
                file_bytes = np.asarray(
                    bytearray(uploaded_file.read()),
                    dtype=np.uint8,
                )
                image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

                if image is None:
                    stats["invalid"] += 1
                    processing_details.append(
                        f"Cannot read image: {uploaded_file.name}"
                    )
                    continue

                faces, _ = detect_faces(image)

                if len(faces) == 0:
                    stats["no_face"] += 1
                    processing_details.append(
                        f"No face detected in file: {uploaded_file.name}"
                    )
                    continue

                if len(faces) > 1:
                    stats["multiple_faces"] += 1
                    processing_details.append(
                        f"Multiple faces detected in file: {uploaded_file.name}"
                    )
                    continue

                x, y, w, h = faces[0]
                face_crop = crop_and_resize_face(image, (x, y, w, h))
                save_face_image(selected_employee_code, face_crop)
                stats["saved"] += 1
                processing_details.append(f"Saved: {uploaded_file.name}")

            result_col_1, result_col_2, result_col_3, result_col_4 = st.columns(4)
            result_col_1.metric("Saved", stats["saved"])
            result_col_2.metric("No Face", stats["no_face"])
            result_col_3.metric("Multiple Faces", stats["multiple_faces"])
            result_col_4.metric("Invalid", stats["invalid"])

            if stats["saved"]:
                st.toast(f"Saved {stats['saved']} image(s) for {selected_employee_code}.")

            with st.expander("Processing Details"):
                for detail in processing_details:
                    st.write(f"- {detail}")

with tab_camera:
    with st.container(border=True):
        st.subheader("Capture Face From Camera")
        st.caption("Keep exactly one face in the frame before saving.")

        control_col_1, control_col_2, control_col_3 = st.columns([1, 0.8, 1])
        with control_col_1:
            st.selectbox("Choose camera", [0, 1], key=f"{PAGE_KEY}_camera_index")
        with control_col_2:
            run_camera = st.checkbox("Run Camera", key=f"{PAGE_KEY}_run_camera")
        with control_col_3:
            capture_button = st.button(
                "Capture Face",
                type="primary",
                use_container_width=True,
            )

        preview_container = st.container()
        if not run_camera:
            release_camera(st.session_state, PAGE_KEY)
            preview_container.info("Turn on 'Run Camera' to start live preview.")

        with preview_container:
            render_live_camera()

        if capture_button:
            latest_frame = st.session_state.get(f"{PAGE_KEY}_frame")

            if not run_camera or latest_frame is None:
                st.toast("Camera is not running.")
            else:
                latest_faces, _ = detect_faces(latest_frame)

                if len(latest_faces) == 0:
                    st.toast("No face detected. Cannot save.")
                elif len(latest_faces) > 1:
                    st.toast("Multiple faces detected. Please keep exactly one face in frame.")
                else:
                    x, y, w, h = latest_faces[0]
                    face_crop = crop_and_resize_face(latest_frame, (x, y, w, h))
                    save_path = save_face_image(selected_employee_code, face_crop)
                    st.toast(f"Saved: {save_path}")


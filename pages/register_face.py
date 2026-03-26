import cv2
import numpy as np
import streamlit as st

from core.camera_stream import (
    annotate_faces,
    get_or_create_camera,
    release_camera,
    update_detected_faces,
)
from core.face_detect import detect_faces
from core.save_face import crop_and_resize_face, save_face_image
from db.database import get_all_employees, initialize_database

PAGE_KEY = "register_face"

initialize_database()

st.title("Register Face")


@st.fragment(run_every=0.15)
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


def main():
    initialize_database()
    st.title("Register Face")

    employees = get_all_employees()

    if not employees:
        st.warning("No employees found. Please add employees first.")
        st.stop()

    employee_options = {
        f"{employee_code} - {name}": employee_code
        for _, employee_code, name, _ in employees
    }

    selected_employee_label = st.selectbox(
        "Choose employee",
        list(employee_options.keys()),
    )

    selected_employee_code = employee_options[selected_employee_label]

    st.write(f"Selected Employee Code: {selected_employee_code}")

    st.subheader("Option 1: Upload face images")

    uploaded_files = st.file_uploader(
        "Upload one or more images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )

    if uploaded_files and st.button("Process Uploaded Images"):
        saved_count = 0

        for uploaded_file in uploaded_files:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            if image is None:
                st.error(f"Cannot read image: {uploaded_file.name}")
                continue

            faces, _ = detect_faces(image)

            if len(faces) == 0:
                st.warning(f"No face detected in file: {uploaded_file.name}")
                continue

            if len(faces) > 1:
                st.warning(
                    f"Multiple faces detected in file: {uploaded_file.name}. "
                    "Please use an image with exactly one face."
                )
                continue

            x, y, w, h = faces[0]
            face_crop = crop_and_resize_face(image, (x, y, w, h))
            save_path = save_face_image(selected_employee_code, face_crop)

            saved_count += 1
            st.success(f"Saved: {save_path}")

        st.info(f"Total saved images: {saved_count}")

    st.subheader("Option 2: Capture face from camera")

    st.selectbox("Choose camera", [0, 1], key=f"{PAGE_KEY}_camera_index")
    run_camera = st.checkbox("Run Camera", key=f"{PAGE_KEY}_run_camera")
    capture_button = st.button("Capture Face")

    if not run_camera:
        release_camera(st.session_state, PAGE_KEY)
        st.info("Turn on 'Run Camera' to start live preview.")

    render_live_camera()

    if capture_button:
        latest_frame = st.session_state.get(f"{PAGE_KEY}_frame")

        if not run_camera or latest_frame is None:
            st.warning("Camera is not running.")
        else:
            latest_faces, _ = detect_faces(latest_frame)

            if len(latest_faces) == 0:
                st.warning("No face detected. Cannot save.")
            elif len(latest_faces) > 1:
                st.warning("Multiple faces detected. Please keep exactly one face in frame.")
            else:
                x, y, w, h = latest_faces[0]
                face_crop = crop_and_resize_face(latest_frame, (x, y, w, h))
                save_path = save_face_image(selected_employee_code, face_crop)
                st.success(f"Saved: {save_path}")


if __name__ == "__main__":
    main()

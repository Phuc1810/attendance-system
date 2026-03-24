import cv2
import numpy as np
import streamlit as st

from core.camera_stream import annotate_faces, get_or_create_camera, release_camera
from core.face_detect import detect_faces
from core.save_face import crop_and_resize_face, save_face_image
from db.database import connect_db, initialize_database

PAGE_KEY = "register_face"

initialize_database()

st.title("Register Face")


def get_employees():
    with connect_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, name FROM employees ORDER BY id")
        return cursor.fetchall()


# Định nghĩa hàm render độc lập
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

    faces, _ = detect_faces(frame)
    faces_list = [tuple(int(value) for value in face) for face in faces]
    st.session_state[f"{PAGE_KEY}_frame"] = frame.copy()
    st.session_state[f"{PAGE_KEY}_faces"] = faces_list

    annotated_frame = annotate_faces(frame, faces_list)
    frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
    st.image(frame_rgb, channels="RGB")

# Bọc toàn bộ logic luồng chính vào hàm main
def main():
    initialize_database()
    st.title("Register Face")

    employees = get_employees()

    if not employees:
        st.warning("No employees found. Please add employees first.")
        st.stop()

    employee_options = {f"{emp_id} - {name}": emp_id for emp_id, name in employees}

    selected_employee_label = st.selectbox(
        "Choose employee",
        list(employee_options.keys()),
    )

    selected_employee_id = employee_options[selected_employee_label]

    st.write(f"Selected Employee ID: {selected_employee_id}")

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

            x, y, w, h = faces[0]
            face_crop = crop_and_resize_face(image, (x, y, w, h))
            save_path = save_face_image(selected_employee_id, face_crop)

            saved_count += 1
            st.success(f"Saved: {save_path}")

        st.info(f"Total saved images: {saved_count}")

    st.subheader("Option 2: Capture face from camera")

    cam_index = st.selectbox("Choose camera", [0, 1], key=f"{PAGE_KEY}_camera_index")
    run_camera = st.checkbox("Run Camera", key=f"{PAGE_KEY}_run_camera")
    capture_button = st.button("Capture Face")

    if not run_camera:
        release_camera(st.session_state, PAGE_KEY)
        st.info("Turn on 'Run Camera' to start live preview.")

    # Chỉ gọi hàm render khi code được chạy thực sự
    render_live_camera()

    if capture_button:
        latest_frame = st.session_state.get(f"{PAGE_KEY}_frame")
        latest_faces = st.session_state.get(f"{PAGE_KEY}_faces", [])

        if not run_camera or latest_frame is None:
            st.warning("Camera is not running.")
        elif len(latest_faces) == 0:
            st.warning("No face detected. Cannot save.")
        else:
            x, y, w, h = latest_faces[0]
            face_crop = crop_and_resize_face(latest_frame, (x, y, w, h))
            save_path = save_face_image(selected_employee_id, face_crop)
            st.success(f"Saved: {save_path}")

# Điểm neo an toàn
if __name__ == "__main__":
    main()
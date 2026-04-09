import cv2
import streamlit as st

from core.camera_stream import (
    annotate_faces,
    get_or_create_camera,
    release_camera,
    release_inactive_cameras,
    update_detected_faces,
)
from core.face_detect import detect_faces
st.set_page_config(page_title="Attendance Dashboard", layout="wide")

PAGE_KEY = "face_detection"
CAMERA_INTERVAL_SECONDS = 0.15

if f"{PAGE_KEY}_run_camera" not in st.session_state:
    st.session_state[f"{PAGE_KEY}_run_camera"] = False

release_inactive_cameras(st.session_state, PAGE_KEY)


camera_run_every = (
    CAMERA_INTERVAL_SECONDS
    if st.session_state.get(f"{PAGE_KEY}_run_camera", False)
    else None
)

st.title("Face Detection")
st.caption("Technical test page for checking camera input and face detection quality.")

control_col_1, control_col_2, control_col_3 = st.columns([1, 0.8, 1.2], gap="large")
with control_col_1:
    st.selectbox("Choose camera", [0, 1], key=f"{PAGE_KEY}_camera_index")
with control_col_2:
    run_camera = st.checkbox("Run Camera", key=f"{PAGE_KEY}_run_camera")
with control_col_3:
    with st.container(border=True):
        st.caption("Test Goal")
        st.write("Verify that the selected camera can detect one or more faces in realtime.")


@st.fragment(run_every=camera_run_every)
def render_live_camera():
    selected_camera_index = st.session_state.get(f"{PAGE_KEY}_camera_index", 0)
    run_camera = st.session_state.get(f"{PAGE_KEY}_run_camera", False)

    preview_col, info_col = st.columns([1.5, 1], gap="large")
    face_count = 0
    status_label = "Stopped"
    status_message = "Turn on 'Run Camera' to start live preview."

    with preview_col:
        with st.container(border=True):
            st.subheader("Live Preview")
            st.caption("The green boxes show the faces currently detected by the model.")

            if not run_camera:
                release_camera(st.session_state, PAGE_KEY)
                st.info(status_message)
            else:
                cap = get_or_create_camera(
                    st.session_state,
                    PAGE_KEY,
                    selected_camera_index,
                )

                if not cap.isOpened():
                    status_label = "Camera Error"
                    status_message = "Cannot open camera."
                    st.error(status_message)
                    release_camera(st.session_state, PAGE_KEY)
                else:
                    ret, frame = cap.read()
                    if not ret:
                        status_label = "Read Error"
                        status_message = "Cannot read frame from camera."
                        st.error(status_message)
                        release_camera(st.session_state, PAGE_KEY)
                    else:
                        faces_list = update_detected_faces(
                            st.session_state,
                            PAGE_KEY,
                            frame,
                            detect_faces,
                        )
                        face_count = len(faces_list)
                        status_label = "Running"
                        status_message = (
                            "Detection is running. Keep faces visible and inside the frame."
                        )

                        annotated_frame = annotate_faces(frame, faces_list)
                        frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                        st.image(frame_rgb, channels="RGB")

    with info_col:
        with st.container(border=True):
            st.subheader("Detection Status")
            metric_col_1, metric_col_2 = st.columns(2)
            metric_col_1.metric("Camera", selected_camera_index)
            metric_col_2.metric("Faces Detected", face_count)
            st.write(f"**Status:** {status_label}")
            st.caption(status_message)
            st.caption("This page is for technical testing only and does not save any data.")


render_live_camera()


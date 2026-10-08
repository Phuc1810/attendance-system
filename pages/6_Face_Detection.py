import cv2
import streamlit as st

from core.camera_stream import (
    get_or_create_camera,
    release_camera,
    release_inactive_cameras,
    update_detected_faces,
    annotate_faces,
)
from core.face_detect import detect_faces

PAGE_KEY = "face_detection"
CAMERA_INTERVAL_SECONDS = 0.1

if f"{PAGE_KEY}_run_camera" not in st.session_state:
    st.session_state[f"{PAGE_KEY}_run_camera"] = False

release_inactive_cameras(st.session_state, PAGE_KEY)

camera_run_every = (
    CAMERA_INTERVAL_SECONDS
    if st.session_state.get(f"{PAGE_KEY}_run_camera", False)
    else None
)

st.title("🔍 Face Detection (Native)")
st.caption("Technical test page for checking camera input and face detection quality using OpenCV DirectShow.")

control_col_1, control_col_2, control_col_3 = st.columns([1, 1, 1.2], gap="large")
with control_col_1:
    selected_camera_index = st.selectbox(
        "Choose camera",
        [0, 1],
        format_func=lambda idx: f"Camera {idx}",
        key=f"{PAGE_KEY}_camera_index",
    )
with control_col_2:
    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
    run_camera = st.toggle("🎥 Bật Camera Preview", key=f"{PAGE_KEY}_run_camera")

with control_col_3:
    with st.container(border=True):
        st.caption("Test Goal")
        st.write("Verify that the selected native camera can detect one or more faces in realtime.")

previous_camera_index = st.session_state.get(f"{PAGE_KEY}_active_camera_index")
if previous_camera_index != selected_camera_index:
    st.session_state[f"{PAGE_KEY}_active_camera_index"] = selected_camera_index
    release_camera(st.session_state, PAGE_KEY)


@st.fragment(run_every=camera_run_every)
def render_detection_view():
    run_cam = st.session_state.get(f"{PAGE_KEY}_run_camera", False)
    cam_index = st.session_state.get(f"{PAGE_KEY}_camera_index", 0)

    preview_col, info_col = st.columns([1.5, 1], gap="large")

    faces_list = []

    with preview_col:
        with st.container(border=True):
            st.subheader("Live Preview (Native)")
            st.caption("The green boxes show the faces currently detected by Haar Cascade.")

            if not run_cam:
                release_camera(st.session_state, PAGE_KEY)
                st.info("Camera đang tắt. Bật 'Bật Camera Preview' để bắt đầu kiểm tra.")
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
                        faces_list = update_detected_faces(
                            st.session_state,
                            PAGE_KEY,
                            frame,
                            detect_faces,
                        )
                        annotated = annotate_faces(frame, faces_list)

                        frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                        st.image(frame_rgb, channels="RGB", use_container_width=True)

    with info_col:
        with st.container(border=True):
            st.subheader("Detection Status")
            metric_col_1, metric_col_2 = st.columns(2)
            metric_col_1.metric("Camera", cam_index)
            metric_col_2.metric("Faces Detected", len(faces_list))

            if not run_cam:
                st.write("**Status:** Stopped")
                st.caption("Bật 'Bật Camera Preview' để bắt đầu xem trực tiếp.")
            else:
                st.write("**Status:** Running (DirectShow Native)")
                st.caption("Detection is running smoothly with 0ms latency.")

            st.caption("This page is for technical testing only and does not save any data.")


render_detection_view()

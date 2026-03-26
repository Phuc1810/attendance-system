#file test nhận khuôn mặt
import cv2
import streamlit as st

from core.camera_stream import (
    annotate_faces,
    get_or_create_camera,
    release_camera,
    update_detected_faces,
)
from core.face_detect import detect_faces

PAGE_KEY = "face_detection"


#Streamlit sẽ tự động chạy lại hàm liên tục 0.15s để tạo hiệu ứng video trực tiếp
@st.fragment(run_every=0.15)
def render_live_camera():
    if not st.session_state.get(f"{PAGE_KEY}_run_camera", False):
        return

    #lấy khung hình cho AI tìm toạ độ
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

    #lấy khung hình cho AI tìm toạ độ mặt
    faces_list = update_detected_faces(
        st.session_state,
        PAGE_KEY,
        frame,
        detect_faces,
    )

    #Vẽ khung xanh và hiển thị số khuôn mặt
    annotated_frame = annotate_faces(frame, faces_list)
    frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
    st.image(frame_rgb, channels="RGB")


def main():
    st.title("Face Detection")

    st.selectbox("Choose camera", [0, 1], key=f"{PAGE_KEY}_camera_index")
    run_camera = st.checkbox("Run camera", key=f"{PAGE_KEY}_run_camera")

    if not run_camera:
        release_camera(st.session_state, PAGE_KEY)
        st.info("Turn on 'Run camera' to start live preview.")

    render_live_camera()


if __name__ == "__main__":
    main()

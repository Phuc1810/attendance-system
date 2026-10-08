import streamlit as st
from pathlib import Path
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from streamlit_autorefresh import st_autorefresh
from core.camera_stream import DetectionProcessor
from core.face_detect import detect_faces
from core.save_face import crop_and_resize_face, save_face_image
from db.database import get_all_employees, initialize_database

# st.set_page_config(page_title="Attendance Dashboard", layout="wide")
st_autorefresh(interval=1000, key="data_refresh")

PAGE_KEY = "register_face"
FACE_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "faces"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

initialize_database()

def count_employee_images(employee_code):
    employee_folder = FACE_DATA_DIR / employee_code
    if not employee_folder.exists():
        return 0
    return sum(
        1 for file_path in employee_folder.iterdir()
        if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS
    )

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
        format_func=lambda code: f"{code} - {employee_map[code]['name']} ({employee_map[code]['department']})",
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
        st.caption("Use one face per image and include multiple head angles for better recognition quality.")
        uploaded_files = st.file_uploader(
            "Upload one or more images",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
        )
        if uploaded_files and st.button("Process Uploaded Images", type="primary", use_container_width=True):
            stats = {"saved": 0, "no_face": 0, "multiple_faces": 0, "invalid": 0}
            processing_details = []
            for uploaded_file in uploaded_files:
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                if image is None:
                    stats["invalid"] += 1
                    processing_details.append(f"Cannot read image: {uploaded_file.name}")
                    continue
                faces, _ = detect_faces(image)
                if len(faces) == 0:
                    stats["no_face"] += 1
                    processing_details.append(f"No face detected in file: {uploaded_file.name}")
                    continue
                if len(faces) > 1:
                    stats["multiple_faces"] += 1
                    processing_details.append(f"Multiple faces detected in file: {uploaded_file.name}")
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
        control_col_1, control_col_3 = st.columns([1, 1])
        with control_col_1:
            st.selectbox("Choose camera", [0, 1], key=f"{PAGE_KEY}_camera_index")
        with control_col_3:
            capture_button = st.button("Capture Face", type="primary", use_container_width=True)

        RTC_CONFIGURATION = RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        )

        ctx = webrtc_streamer(
            key="register_face",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_processor_factory=DetectionProcessor,
            media_stream_constraints={
                "video": {
                    "width": {"ideal": 640},
                    "height": {"ideal": 480}
                },
                "audio": False
            },
            async_processing=True,
        )

        if capture_button:
            if not (ctx and ctx.state.playing and ctx.video_processor and ctx.video_processor.frame_bgr is not None):
                st.toast("Camera is not running or no frame available.")
            else:
                latest_frame = ctx.video_processor.frame_bgr
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

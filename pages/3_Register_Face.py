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

st.title("🧑‍💻 Register Face")
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

tab_camera, tab_upload = st.tabs(["📷 Camera Capture", "📁 Upload Images"])

with tab_camera:
    cam_col, info_col = st.columns([1.5, 1], gap="large")
    with cam_col:
        st.markdown("### Live Preview")
        st.markdown(
            """
            <style>
            .stVideo { border-radius: 12px; overflow: hidden; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); }
            </style>
            """, 
            unsafe_allow_html=True
        )

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

    with info_col:
        st.markdown("### Capture Controls")
        st.caption("Step 1. Start the camera and look straight.")
        st.caption("Step 2. Keep exactly one face in the frame.")
        st.caption("Step 3. Click Capture.")
        
        st.selectbox("Camera Source", [0, 1], key=f"{PAGE_KEY}_camera_index", disabled=True, help="Browser manages camera source.")
        
        st.markdown("<br>", unsafe_allow_html=True)
        capture_button = st.button("📸 Capture Face", type="primary", use_container_width=True)

        if capture_button:
            if not (ctx and ctx.state.playing and ctx.video_processor and ctx.video_processor.frame_bgr is not None):
                st.toast("⚠️ Camera is not running or no frame available.")
            else:
                latest_frame = ctx.video_processor.frame_bgr
                latest_faces, _ = detect_faces(latest_frame)
                if len(latest_faces) == 0:
                    st.toast("⚠️ No face detected. Cannot save.")
                elif len(latest_faces) > 1:
                    st.toast("⚠️ Multiple faces detected. Keep exactly one face in frame.")
                else:
                    x, y, w, h = latest_faces[0]
                    face_crop = crop_and_resize_face(latest_frame, (x, y, w, h))
                    save_path = save_face_image(selected_employee_code, face_crop)
                    st.toast(f"✅ Saved face image!")


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

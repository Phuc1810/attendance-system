import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from streamlit_autorefresh import st_autorefresh
from core.camera_stream import DetectionProcessor

# st.set_page_config(page_title="Attendance Dashboard", layout="wide")
st_autorefresh(interval=1000, key="data_refresh")

st.title("Face Detection")
st.caption("Technical test page for checking camera input and face detection quality.")

control_col_1, control_col_3 = st.columns([1, 1.2], gap="large")
with control_col_1:
    st.selectbox("Choose camera", [0, 1], key="face_detection_camera_index")
with control_col_3:
    with st.container(border=True):
        st.caption("Test Goal")
        st.write("Verify that the selected camera can detect one or more faces in realtime.")

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

preview_col, info_col = st.columns([1.5, 1], gap="large")

with preview_col:
    with st.container(border=True):
        st.subheader("Live Preview")
        st.caption("The green boxes show the faces currently detected by the model.")
        
        ctx = webrtc_streamer(
            key="face_detection",
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
    with st.container(border=True):
        st.subheader("Detection Status")
        metric_col_1, metric_col_2 = st.columns(2)
        metric_col_1.metric("Camera", st.session_state.get("face_detection_camera_index", 0))
        
        if ctx and ctx.state.playing and ctx.video_processor:
            faces = ctx.video_processor.faces_list
            metric_col_2.metric("Faces Detected", len(faces))
            st.write(f"**Status:** Running")
            st.caption("Detection is running. Keep faces visible and inside the frame.")
        else:
            metric_col_2.metric("Faces Detected", 0)
            st.write(f"**Status:** Stopped")
            st.caption("Turn on 'START' to begin live preview.")
            
        st.caption("This page is for technical testing only and does not save any data.")

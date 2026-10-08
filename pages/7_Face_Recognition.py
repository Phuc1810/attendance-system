import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from streamlit_autorefresh import st_autorefresh
from core.camera_stream import RecognitionProcessor
from core.face_recognizer import train_model
from db.database import initialize_database

st.set_page_config(page_title="Attendance Dashboard", layout="wide")
st_autorefresh(interval=1000, key="data_refresh")

PAGE_KEY = "face_recognition"
initialize_database()

st.title("Face Recognition")
st.caption("Train the recognition model, then test how well it identifies faces in realtime.")

train_col, summary_col = st.columns([1.1, 1], gap="large")

with train_col:
    with st.container(border=True):
        st.subheader("Step 1: Train Model")
        st.caption("Retrain after adding or deleting employees or face images so the model stays up to date.")

        if st.button("Train Recognition Model", type="primary", use_container_width=True):
            try:
                result = train_model()
                st.session_state[f"{PAGE_KEY}_train_result"] = result
                st.session_state[f"{PAGE_KEY}_train_error"] = None
            except Exception as error:
                st.session_state[f"{PAGE_KEY}_train_result"] = None
                st.session_state[f"{PAGE_KEY}_train_error"] = str(error)

with summary_col:
    with st.container(border=True):
        st.subheader("Training Summary")
        train_error = st.session_state.get(f"{PAGE_KEY}_train_error")
        train_result = st.session_state.get(f"{PAGE_KEY}_train_result")

        if train_error:
            st.error(f"Training failed: {train_error}")
        elif train_result:
            metric_col_1, metric_col_2 = st.columns(2)
            metric_col_1.metric("Images", train_result["num_images"])
            metric_col_2.metric("People", train_result["num_people"])

            with st.expander("Employee labels used in the model"):
                st.json(train_result["label_to_code"])

            with st.expander("Unknown rejection thresholds"):
                st.json(train_result["code_thresholds"])

            with st.expander("Camera-aware threshold profiles"):
                st.json(train_result["camera_profiles"])
        else:
            st.info("Train the model once to view the summary and current thresholds.")

st.subheader("Step 2: Test Recognition")
control_col_1, control_col_3 = st.columns([1, 1.2], gap="large")
with control_col_1:
    selected_camera_index = st.selectbox("Choose camera", [0, 1], key=f"{PAGE_KEY}_camera_index")
with control_col_3:
    with st.container(border=True):
        st.caption("Test Goal")
        st.write("Use this page to verify whether the trained model returns a correct employee code or Unknown.")

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

preview_col, status_col = st.columns([1.5, 1], gap="large")

with preview_col:
    with st.container(border=True):
        st.subheader("Live Recognition")
        st.caption("Keep exactly one face visible if you want a clean recognition result.")
        
        ctx = webrtc_streamer(
            key="face_recognition",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_processor_factory=RecognitionProcessor,
            media_stream_constraints={
                "video": {
                    "width": {"ideal": 640},
                    "height": {"ideal": 480}
                },
                "audio": False
            },
            async_processing=True,
        )

        if ctx and ctx.video_processor:
            ctx.video_processor.camera_index = selected_camera_index

with status_col:
    with st.container(border=True):
        st.subheader("Recognition Status")
        metric_col_1, metric_col_2 = st.columns(2)
        metric_col_1.metric("Camera", selected_camera_index)
        
        face_count = 0
        if ctx and ctx.state.playing and ctx.video_processor:
            face_count = len(ctx.video_processor.faces_list)
            
        metric_col_2.metric("Faces", face_count)
        
        if not (ctx and ctx.state.playing):
            st.write(f"**Status:** Stopped")
            st.caption("Turn on 'START' to begin recognition.")
            st.info("No recognition result available yet.")
        elif ctx.video_processor:
            if face_count == 0:
                st.write("**Status:** No Face")
                st.caption("No face detected in the current frame.")
            elif face_count > 1:
                st.write("**Status:** Multiple Faces")
                st.caption("Recognition works best when only one face is visible.")
            else:
                st.write("**Status:** Running")
                st.caption("One face detected. Recognition result is shown below.")
                
            pred = ctx.video_processor.prediction
            if pred:
                if pred["is_match"]:
                    st.success(f"Matched employee: {pred['display_code']}")
                    st.write(f"**Confidence:** {pred['confidence']:.2f}")
                    st.write(f"**Threshold:** {pred['match_threshold']:.2f}")
                else:
                    st.warning("Recognition result: Unknown")
                    st.write(f"**Confidence:** {pred['confidence']:.2f}")
                    st.write(f"**Threshold:** {pred['match_threshold']:.2f}")

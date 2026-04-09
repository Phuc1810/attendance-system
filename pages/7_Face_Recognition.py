import cv2
import streamlit as st
st.set_page_config(page_title="Attendance Dashboard", layout="wide")

from core.camera_stream import (
    get_or_create_camera,
    get_or_update_prediction,
    release_camera,
    release_inactive_cameras,
    update_detected_faces,
)
from core.face_detect import detect_faces
from core.face_recognizer import predict_face, train_model
from core.save_face import crop_and_resize_face
from db.database import initialize_database

PAGE_KEY = "face_recognition"
CAMERA_INTERVAL_SECONDS = 0.15
RECOGNITION_PADDING_RATIO = 0.0

initialize_database()

if f"{PAGE_KEY}_run_camera" not in st.session_state:
    st.session_state[f"{PAGE_KEY}_run_camera"] = False

release_inactive_cameras(st.session_state, PAGE_KEY)

camera_run_every = (
    CAMERA_INTERVAL_SECONDS
    if st.session_state.get(f"{PAGE_KEY}_run_camera", False)
    else None
)

st.title("Face Recognition")
st.caption("Train the recognition model, then test how well it identifies faces in realtime.")

train_col, summary_col = st.columns([1.1, 1], gap="large")

with train_col:
    with st.container(border=True):
        st.subheader("Step 1: Train Model")
        st.caption(
            "Retrain after adding or deleting employees or face images so the model stays up to date."
        )

        if st.button(
            "Train Recognition Model",
            type="primary",
            use_container_width=True,
        ):
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
control_col_1, control_col_2, control_col_3 = st.columns([1, 0.8, 1.2], gap="large")
with control_col_1:
    st.selectbox("Choose camera", [0, 1], key=f"{PAGE_KEY}_camera_index")
with control_col_2:
    run_camera = st.checkbox("Run Camera", key=f"{PAGE_KEY}_run_camera")
with control_col_3:
    with st.container(border=True):
        st.caption("Test Goal")
        st.write("Use this page to verify whether the trained model returns a correct employee code or Unknown.")


@st.fragment(run_every=camera_run_every)
def render_live_camera():
    selected_camera_index = st.session_state.get(f"{PAGE_KEY}_camera_index", 0)
    run_camera = st.session_state.get(f"{PAGE_KEY}_run_camera", False)

    preview_col, status_col = st.columns([1.5, 1], gap="large")
    face_count = 0
    status_label = "Stopped"
    status_message = "Turn on 'Run Camera' to start recognition."
    matched_code = None
    matched_confidence = None
    match_threshold = None

    with preview_col:
        with st.container(border=True):
            st.subheader("Live Recognition")
            st.caption("Keep exactly one face visible if you want a clean recognition result.")

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
                        annotated_frame = frame.copy()
                        prediction = None
                        model_error = False

                        if face_count == 0:
                            status_label = "No Face"
                            status_message = "No face detected in the current frame."
                        elif face_count > 1:
                            status_label = "Multiple Faces"
                            status_message = "Recognition works best when only one face is visible."
                        else:
                            status_label = "Running"
                            status_message = "One face detected. Recognition result is shown below."

                        try:
                            prediction, _ = get_or_update_prediction(
                                st.session_state,
                                PAGE_KEY,
                                frame,
                                faces_list,
                                crop_and_resize_face,
                                predict_face,
                                camera_index=selected_camera_index,
                                padding_ratio=RECOGNITION_PADDING_RATIO,
                            )
                        except Exception:
                            prediction = None
                            model_error = True
                            status_label = "Model Error"
                            status_message = "The recognition model is not ready or failed to load."

                        for (x, y, w, h) in faces_list:
                            if face_count == 1 and prediction is not None:
                                matched_confidence = prediction["confidence"]
                                match_threshold = prediction["match_threshold"]

                                if prediction["is_match"]:
                                    matched_code = prediction["display_code"]
                                    label_text = (
                                        f"{prediction['display_code']} "
                                        f"({prediction['confidence']:.2f})"
                                    )
                                    box_color = (0, 255, 0)
                                else:
                                    label_text = (
                                        f"Unknown ({prediction['confidence']:.2f} > "
                                        f"{prediction['match_threshold']:.2f})"
                                    )
                                    box_color = (0, 0, 255)
                            elif model_error:
                                label_text = "Model error"
                                box_color = (0, 0, 255)
                            else:
                                label_text = "Face detected"
                                box_color = (0, 215, 255)

                            cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), box_color, 2)
                            cv2.putText(
                                annotated_frame,
                                label_text,
                                (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8,
                                box_color,
                                2,
                            )

                        cv2.putText(
                            annotated_frame,
                            f"Faces detected: {face_count}",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            2,
                        )

                        frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                        st.image(frame_rgb, channels="RGB")

    with status_col:
        with st.container(border=True):
            st.subheader("Recognition Status")
            metric_col_1, metric_col_2 = st.columns(2)
            metric_col_1.metric("Camera", selected_camera_index)
            metric_col_2.metric("Faces", face_count)
            st.write(f"**Status:** {status_label}")
            st.caption(status_message)

            if matched_code:
                st.success(f"Matched employee: {matched_code}")
                st.write(f"**Confidence:** {matched_confidence:.2f}")
                st.write(f"**Threshold:** {match_threshold:.2f}")
            elif matched_confidence is not None:
                st.warning("Recognition result: Unknown")
                st.write(f"**Confidence:** {matched_confidence:.2f}")
                st.write(f"**Threshold:** {match_threshold:.2f}")
            else:
                st.info("No recognition result available yet.")


render_live_camera()

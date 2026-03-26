import cv2
import streamlit as st

from core.camera_stream import (
    annotate_faces,
    get_or_create_camera,
    release_camera,
    update_detected_faces,
)
from core.face_detect import detect_faces
from core.face_recognizer import predict_face, train_model
from core.save_face import crop_and_resize_face
from db.database import initialize_database

PAGE_KEY = "face_recognition"

initialize_database()

st.title("Face Recognition")


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

    annotated_frame = frame.copy()

    for (x, y, w, h) in faces_list:
        cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        face_crop = crop_and_resize_face(frame, (x, y, w, h))

        try:
            predicted_code, confidence = predict_face(face_crop)

            # Ngưỡng demo ban đầu: confidence thấp thì chấp nhận
            if confidence < 80:
                label_text = f"{predicted_code} ({confidence:.2f})"
            else:
                label_text = f"Unknown ({confidence:.2f})"

        except Exception as error:
            label_text = f"Model error"

        cv2.putText(
            annotated_frame,
            label_text,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

    cv2.putText(
        annotated_frame,
        f"Faces detected: {len(faces_list)}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )

    frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
    st.image(frame_rgb, channels="RGB")


def main():
    st.subheader("Step 1: Train model")

    if st.button("Train Recognition Model"):
        try:
            result = train_model()
            st.success(
                f"Training complete! "
                f"Images: {result['num_images']} | "
                f"People: {result['num_people']}"
            )
            st.json(result["label_to_code"])
        except Exception as error:
            st.error(f"Training failed: {error}")

    st.subheader("Step 2: Test recognition with camera")

    st.selectbox("Choose camera", [0, 1], key=f"{PAGE_KEY}_camera_index")
    run_camera = st.checkbox("Run Camera", key=f"{PAGE_KEY}_run_camera")

    if not run_camera:
        release_camera(st.session_state, PAGE_KEY)
        st.info("Turn on 'Run Camera' to start recognition.")

    render_live_camera()


if __name__ == "__main__":
    main()
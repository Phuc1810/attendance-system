import cv2


def get_or_create_camera(session_state, prefix, camera_index):
    cap_key = f"{prefix}_cap"
    index_key = f"{prefix}_camera_index_value"
    cap = session_state.get(cap_key)
    current_index = session_state.get(index_key)

    if cap is None or current_index != camera_index or not cap.isOpened():
        if cap is not None:
            cap.release()
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        session_state[cap_key] = cap
        session_state[index_key] = camera_index

    return cap


def release_camera(session_state, prefix):
    cap_key = f"{prefix}_cap"
    cap = session_state.get(cap_key)
    if cap is not None:
        cap.release()
    session_state[cap_key] = None
    session_state[f"{prefix}_frame"] = None
    session_state[f"{prefix}_faces"] = []


def annotate_faces(frame, faces):
    annotated = frame.copy()

    for (x, y, w, h) in faces:
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.putText(
        annotated,
        f"Faces detected: {len(faces)}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )

    return annotated

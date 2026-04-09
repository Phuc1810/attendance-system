import cv2

DETECTION_INTERVAL = 3
PREDICTION_INTERVAL = 2
CAMERA_FRAME_WIDTH = 640
CAMERA_FRAME_HEIGHT = 480
PREDICTION_POSITION_TOLERANCE = 18
PREDICTION_SIZE_TOLERANCE = 18
CAMERA_PAGE_KEYS = ("register_face", "attendance", "face_detection", "face_recognition")


# Quan ly viec bat, tat camera va toi uu hoa ve hinh tren Streamlit
# Dam bao tai cung 1 thoi diem chi co 1 cam co the hoat dong

def get_or_create_camera(session_state, prefix, camera_index):
    cap_key = f"{prefix}_cap"
    index_key = f"{prefix}_camera_index_value"
    cap = session_state.get(cap_key)
    current_index = session_state.get(index_key)

    if cap is None or current_index != camera_index or not cap.isOpened():
        if cap is not None:
            cap.release()
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_FRAME_HEIGHT)
        session_state[cap_key] = cap
        session_state[index_key] = camera_index

    return cap


# Giai phong tai nguyen tranh giu session webcam

def clear_prediction_cache(session_state, prefix):
    session_state[f"{prefix}_prediction"] = None
    session_state[f"{prefix}_prediction_face"] = None
    session_state[f"{prefix}_prediction_counter"] = 0


def release_camera(session_state, prefix):
    cap_key = f"{prefix}_cap"
    cap = session_state.get(cap_key)
    if cap is not None:
        cap.release()
    session_state[cap_key] = None
    session_state[f"{prefix}_frame"] = None
    session_state[f"{prefix}_faces"] = []
    session_state[f"{prefix}_frame_counter"] = 0
    clear_prediction_cache(session_state, prefix)


# Ham nay co muc dich la luu frame moi nhat, nho ket qua detect truoc do
# va chi thuc hien face_detect theo chu ky

def update_detected_faces(session_state, prefix, frame, detect_faces_fn):
    frame_key = f"{prefix}_frame"
    faces_key = f"{prefix}_faces"
    frame_counter_key = f"{prefix}_frame_counter"

    frame_counter = session_state.get(frame_counter_key, 0) + 1
    session_state[frame_counter_key] = frame_counter
    session_state[frame_key] = frame.copy()

    previous_faces = session_state.get(faces_key, [])
    should_detect = (
        frame_counter == 1
        or frame_counter % DETECTION_INTERVAL == 0
        or not previous_faces
    )

    if should_detect:
        detected_faces, _ = detect_faces_fn(frame)
        faces_list = [tuple(int(value) for value in face) for face in detected_faces]
        session_state[faces_key] = faces_list
        return faces_list

    return previous_faces


def _normalize_face_box(face_box):
    return tuple(int(value) for value in face_box)


def _face_box_changed(previous_face_box, current_face_box):
    if previous_face_box is None:
        return True

    previous_x, previous_y, previous_w, previous_h = previous_face_box
    current_x, current_y, current_w, current_h = current_face_box

    return (
        abs(previous_x - current_x) > PREDICTION_POSITION_TOLERANCE
        or abs(previous_y - current_y) > PREDICTION_POSITION_TOLERANCE
        or abs(previous_w - current_w) > PREDICTION_SIZE_TOLERANCE
        or abs(previous_h - current_h) > PREDICTION_SIZE_TOLERANCE
    )


def get_or_update_prediction(
    session_state,
    prefix,
    frame,
    faces_list,
    crop_face_fn,
    predict_face_fn,
    camera_index=None,
    padding_ratio=0.0,
):
    if len(faces_list) != 1:
        clear_prediction_cache(session_state, prefix)
        return None, False

    prediction_key = f"{prefix}_prediction"
    prediction_face_key = f"{prefix}_prediction_face"
    prediction_counter_key = f"{prefix}_prediction_counter"

    current_face_box = _normalize_face_box(faces_list[0])
    previous_face_box = session_state.get(prediction_face_key)
    cached_prediction = session_state.get(prediction_key)

    prediction_counter = session_state.get(prediction_counter_key, 0) + 1
    session_state[prediction_counter_key] = prediction_counter

    should_refresh_prediction = (
        cached_prediction is None
        or prediction_counter == 1
        or prediction_counter % PREDICTION_INTERVAL == 0
        or _face_box_changed(previous_face_box, current_face_box)
    )

    if should_refresh_prediction:
        face_crop = crop_face_fn(
            frame,
            current_face_box,
            padding_ratio=padding_ratio,
        )

        prediction_kwargs = {}
        if camera_index is not None:
            prediction_kwargs["camera_index"] = camera_index

        try:
            prediction = predict_face_fn(face_crop, **prediction_kwargs)
        except Exception:
            clear_prediction_cache(session_state, prefix)
            raise

        session_state[prediction_key] = prediction
        session_state[prediction_face_key] = current_face_box
        return prediction, True

    return cached_prediction, False


def release_inactive_cameras(session_state, active_prefix=None):
    for prefix in CAMERA_PAGE_KEYS:
        if active_prefix is not None and prefix == active_prefix:
            continue

        release_camera(session_state, prefix)
        run_key = f"{prefix}_run_camera"
        if run_key in session_state:
            session_state[run_key] = False


# Ve khung va hien thi thong tin

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

import cv2

DETECTION_INTERVAL = 3

#Quản lý việc bật, tắt camera và tối ưu hoá vẽ hình trên Streamlit

#Đảm bảo tại cùng 1 thời điểm chỉ có 1 cam có thể hoạt động
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


# Giải phóng tài nguyên tránh giữ session webcam
def release_camera(session_state, prefix):
    cap_key = f"{prefix}_cap"
    cap = session_state.get(cap_key)
    if cap is not None:
        cap.release()
    session_state[cap_key] = None
    session_state[f"{prefix}_frame"] = None
    session_state[f"{prefix}_faces"] = []
    session_state[f"{prefix}_frame_counter"] = 0


#Hàm này có mục đích là lưu frame mới nhất, nhớ kết quả detect trước đó
# và chỉ thực hiện face_detect theo chu kỳ
def update_detected_faces(session_state, prefix, frame, detect_faces_fn):
    #tên key động trong session_state để phân chia các vùng nhớ
    frame_key = f"{prefix}_frame"
    faces_key = f"{prefix}_faces"
    frame_counter_key = f"{prefix}_frame_counter"

    #Mỗi lần cam đọc được frame mới thì bộ đếm tăng 1
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

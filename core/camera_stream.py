import cv2
from typing import Optional, Tuple, List, Dict, Any

CAMERA_FRAME_WIDTH = 640
CAMERA_FRAME_HEIGHT = 480
CAMERA_PAGE_KEYS = ("register_face", "attendance", "face_detection", "face_recognition")


def get_or_create_camera(session_state: dict, prefix: str, camera_index: int) -> Optional[cv2.VideoCapture]:
    """
    Khởi tạo hoặc tái sử dụng camera phần cứng native qua Windows DirectShow (cv2.CAP_DSHOW).
    Thiết lập buffer size = 1 để loại bỏ hoàn toàn độ trễ tích lũy khung hình.
    """
    cap_key = f"{prefix}_cap"
    index_key = f"{prefix}_camera_index_value"
    cap = session_state.get(cap_key)
    current_index = session_state.get(index_key)

    if cap is None or current_index != camera_index or not cap.isOpened():
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
        
        # Mở camera bằng backend DirectShow của Windows để tối ưu tốc độ và không delay
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            # Fallback sang default backend nếu DirectShow không hỗ trợ thiết bị này
            cap = cv2.VideoCapture(camera_index)

        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_FRAME_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_FRAME_HEIGHT)
            # Quan trọng: Đặt buffer size = 1 để DirectShow không lưu hàng đợi frame cũ
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            session_state[cap_key] = cap
            session_state[index_key] = camera_index
        else:
            session_state[cap_key] = None
            session_state[index_key] = None
            return None

    return cap


def read_camera_frame(cap: cv2.VideoCapture) -> Tuple[bool, Optional[Any]]:
    """
    Đọc 1 khung hình từ thiết bị camera đã mở.
    """
    if cap is None or not cap.isOpened():
        return False, None
    ret, frame = cap.read()
    return ret, frame


def release_camera(session_state: dict, prefix: str) -> None:
    """
    Giải phóng camera phần cứng và dọn dẹp các biến liên quan trong session_state.
    """
    cap_key = f"{prefix}_cap"
    cap = session_state.get(cap_key)
    if cap is not None:
        try:
            cap.release()
        except Exception:
            pass
    session_state[cap_key] = None
    session_state[f"{prefix}_camera_index_value"] = None
    session_state[f"{prefix}_run_camera"] = False


def release_inactive_cameras(session_state: dict, active_prefix: Optional[str] = None) -> None:
    """
    Đóng tất cả các camera đang mở của các trang khác để tránh lỗi xung đột phần cứng
    (Windows chỉ cho phép 1 tiến trình/tab giữ quyền điều khiển webcam tại một thời điểm).
    """
    for prefix in CAMERA_PAGE_KEYS:
        if active_prefix is not None and prefix == active_prefix:
            continue
        release_camera(session_state, prefix)


DETECTION_INTERVAL = 2  # Chỉ chạy Haar Cascade 1 lần mỗi 2 frames, frame xen kẽ dùng lại box cũ


def update_detected_faces(session_state: dict, prefix: str, frame: Any, detect_faces_fn) -> List[Tuple[int, int, int, int]]:
    """
    Tối ưu hóa tốc độ nhận diện bằng cách nhảy cóc theo DETECTION_INTERVAL.
    Giúp đẩy FPS lên mượt mà (25-30 FPS) mà không tốn CPU quét liên tục.
    """
    faces_key = f"{prefix}_faces"
    counter_key = f"{prefix}_detect_counter"

    counter = session_state.get(counter_key, 0) + 1
    session_state[counter_key] = counter
    previous_faces = session_state.get(faces_key, [])

    if counter == 1 or counter % DETECTION_INTERVAL == 0 or not previous_faces:
        detected_faces, _ = detect_faces_fn(frame)
        faces_list = [tuple(int(v) for v in face) for face in detected_faces]
        session_state[faces_key] = faces_list
        return faces_list

    return previous_faces


def annotate_faces(frame: Any, faces: List[Tuple[int, int, int, int]]) -> Any:
    """
    Vẽ khung nhận diện khuôn mặt cơ bản (Detection view).
    """
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


def annotate_recognition(
    frame: Any,
    faces: List[Tuple[int, int, int, int]],
    prediction: Optional[Dict[str, Any]] = None,
    model_error: bool = False,
) -> Any:
    """
    Vẽ khung nhận diện kèm thông tin nhân viên hoặc cảnh báo Unknown.
    """
    annotated = frame.copy()
    for (x, y, w, h) in faces:
        if len(faces) == 1 and prediction is not None:
            if prediction.get("is_match", False):
                label_text = f"{prediction['display_code']} ({prediction['confidence']:.2f})"
                box_color = (0, 255, 0)
            else:
                label_text = f"Unknown ({prediction['confidence']:.2f} > {prediction['match_threshold']:.2f})"
                box_color = (0, 0, 255)
        elif model_error:
            label_text = "Model error"
            box_color = (0, 0, 255)
        else:
            label_text = "Face detected"
            box_color = (0, 215, 255)

        cv2.rectangle(annotated, (x, y), (x + w, y + h), box_color, 2)
        cv2.putText(
            annotated,
            label_text,
            (x, max(20, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            box_color,
            2,
        )

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


# --- STUB CHO CÁC CLASS CŨ ĐỂ ĐẢM BẢO TƯƠNG THÍCH TRƯỚC KHI REFACTOR PHASE 2/3 ---
try:
    from streamlit_webrtc import VideoProcessorBase
    import av

    class DetectionProcessor(VideoProcessorBase):
        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            return frame

    class RecognitionProcessor(VideoProcessorBase):
        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            return frame

    class AttendanceProcessor(VideoProcessorBase):
        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            return frame
except ImportError:
    pass

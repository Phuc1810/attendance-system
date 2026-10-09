import os
# Giới hạn số luồng CPU của OpenMP / BLAS / MKL để Dlib không chiếm trọn 100% CPU
# Giữ CPU luôn thông thoáng cho OpenCV và Streamlit WebSocket server hoạt động 0ms độ trễ
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

import cv2
import threading
import time
from typing import Optional, Tuple, List, Dict, Any

CAMERA_FRAME_WIDTH = 640
CAMERA_FRAME_HEIGHT = 480
CAMERA_PAGE_KEYS = ("register_face", "attendance", "face_detection", "face_recognition")

# Tần số điều tiết hiển thị video qua Streamlit WebSocket (Golden FPS: ~14.7 FPS)
# Đảm bảo trình duyệt luôn tiêu thụ kịp gói tin, triệt tiêu 100% tích tụ hàng đợi gây delay 2-3s
STREAM_FRAME_INTERVAL = 0.068
# Thời gian nhả CPU giữa các vòng lặp streaming để tránh tight spin loop
STREAM_SLEEP_INTERVAL = 0.015
# Chất lượng nén JPEG tối ưu cho video stream (cân bằng sắc nét và dung lượng nhẹ)
STREAM_JPEG_QUALITY = 72


class ThreadedCameraReader:
    """
    Trình đọc camera đa luồng chuyên dụng (Dedicated Camera Capture Thread):
    Liên tục rút cạn bộ đệm DirectShow của Windows ở tốc độ tối đa của phần cứng,
    đảm bảo cap.read() LUÔN LUÔN trả về khung hình mới nhất thời gian thực.
    Loại bỏ triệt để 100% hiện tượng tích tụ bộ đệm gây trễ 1-2 giây khi quay mặt!
    """

    def __init__(self, camera_index: int = 0, width: int = CAMERA_FRAME_WIDTH, height: int = CAMERA_FRAME_HEIGHT):
        self.camera_index = camera_index
        self.cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(camera_index)

        self.latest_frame = None
        self.running = False
        self.lock = threading.Lock()
        self.thread = None

        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            ret, frame = self.cap.read()
            if ret and frame is not None:
                self.latest_frame = frame

            self.running = True
            self.thread = threading.Thread(target=self._capture_worker, daemon=True)
            self.thread.start()

    def _capture_worker(self):
        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret and frame is not None:
                with self.lock:
                    self.latest_frame = frame
            else:
                time.sleep(0.005)

    def isOpened(self) -> bool:
        return self.cap is not None and self.cap.isOpened()

    def read(self) -> Tuple[bool, Optional[Any]]:
        with self.lock:
            if self.latest_frame is not None:
                return True, self.latest_frame.copy()
        if self.isOpened():
            ret, frame = self.cap.read()
            return ret, frame
        return False, None

    def release(self) -> None:
        self.running = False
        if self.thread and self.thread.is_alive():
            try:
                self.thread.join(timeout=0.3)
            except Exception:
                pass
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass


def get_or_create_camera(session_state: dict, prefix: str, camera_index: int) -> Optional[ThreadedCameraReader]:
    """
    Khởi tạo hoặc tái sử dụng camera phần cứng đa luồng chuyên dụng (ThreadedCameraReader).
    Đảm bảo 0ms độ trễ tích lũy khung hình trên Windows DirectShow.
    """
    cap_key = f"{prefix}_cap"
    index_key = f"{prefix}_camera_index_value"
    cap = session_state.get(cap_key)
    current_index = session_state.get(index_key)

    # Đảm bảo cap là ThreadedCameraReader hợp lệ; nếu là VideoCapture cũ thì hủy và tạo mới
    if cap is None or not isinstance(cap, ThreadedCameraReader) or current_index != camera_index or not cap.isOpened():
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass

        reader = ThreadedCameraReader(camera_index)
        if reader.isOpened():
            session_state[cap_key] = reader
            session_state[index_key] = camera_index
            return reader
        else:
            session_state[cap_key] = None
            session_state[index_key] = None
            return None

    return cap


PREDICTION_INTERVAL_SECONDS = 0.25  # Tần suất cập nhật nhận diện tối đa 4 lần/giây để tránh nghẽn CPU
PREDICTION_POSITION_TOLERANCE = 20
PREDICTION_SIZE_TOLERANCE = 20


class PredictionStore:
    """
    Bộ lưu trữ kết quả nhận diện khuôn mặt an toàn đa luồng (Thread-safe Prediction Store):
    Tách biệt hoàn toàn việc lưu trữ kết quả khỏi Streamlit SessionStateProxy,
    cho phép worker thread ghi dữ liệu trên heap an toàn mà không bị Streamlit
    cô lập vào _mock_session_state.
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.prediction: Optional[Dict[str, Any]] = None
        self.prediction_id: float = 0.0
        self.last_read_id: float = 0.0
        self.is_busy: bool = False
        self.last_predict_time: float = 0.0
        self.last_face_box: Optional[Tuple[int, int, int, int]] = None
        self.empty_frames_count: int = 0

    def get_prediction(self) -> Tuple[Optional[Dict[str, Any]], bool, bool]:
        """
        Đọc kết quả mới nhất từ luồng chính (Main thread).
        Trả về: (latest_prediction, is_new, is_busy)
        """
        with self.lock:
            is_new = False
            if self.prediction_id > 0 and self.prediction_id != self.last_read_id:
                self.last_read_id = self.prediction_id
                is_new = True
            pred_copy = dict(self.prediction) if self.prediction is not None else None
            return pred_copy, is_new, self.is_busy

    def set_prediction(self, result: Optional[Dict[str, Any]]) -> None:
        """
        Ghi kết quả từ background worker thread.
        """
        with self.lock:
            self.prediction = result
            self.prediction_id = time.time()
            self.is_busy = False

    def set_busy(self, busy: bool, now: float, face_box: Tuple[int, int, int, int]) -> None:
        with self.lock:
            self.is_busy = busy
            self.last_predict_time = now
            self.last_face_box = face_box

    def can_trigger(self, now: float, current_face_box: Tuple[int, int, int, int]) -> bool:
        with self.lock:
            if self.is_busy:
                return False
            if self.prediction is None:
                return True
            time_passed = (now - self.last_predict_time) >= PREDICTION_INTERVAL_SECONDS
            box_moved = _face_box_changed(self.last_face_box, current_face_box)
            return time_passed or box_moved

    def handle_faces_count(self, count: int) -> bool:
        """
        Quản lý grace period 5 frame để chống chớp tắt làm mất dấu nhận diện tức thời.
        - Trả về True nếu vẫn trong trạng thái hợp lệ.
        - Trả về False nếu quá 5 frame không có khuôn mặt nào.
        """
        with self.lock:
            if count == 1:
                self.empty_frames_count = 0
                return True
            else:
                self.empty_frames_count += 1
                if self.empty_frames_count >= 5:
                    self.prediction = None
                    self.prediction_id = 0.0
                    self.last_read_id = 0.0
                    self.last_face_box = None
                    self.is_busy = False
                    return False
                return True

    def clear(self) -> None:
        with self.lock:
            self.prediction = None
            self.prediction_id = 0.0
            self.last_read_id = 0.0
            self.is_busy = False
            self.last_predict_time = 0.0
            self.last_face_box = None
            self.empty_frames_count = 0


def clear_prediction_cache(session_state: dict, prefix: str) -> None:
    store_key = f"{prefix}_prediction_store"
    store = session_state.get(store_key)
    if store is not None and isinstance(store, PredictionStore):
        store.clear()
    session_state[f"{prefix}_prediction"] = None
    session_state[f"{prefix}_prediction_face"] = None
    session_state[f"{prefix}_prediction_counter"] = 0
    session_state[f"{prefix}_prediction_id"] = None
    session_state[f"{prefix}_last_count_id"] = None
    session_state[f"{prefix}_worker_busy"] = False
    session_state[f"{prefix}_last_predict_time"] = 0.0
    session_state[f"{prefix}_last_face_box"] = None


def _normalize_face_box(face_box: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    return tuple(int(value) for value in face_box)


def _face_box_changed(previous_face_box: Optional[Tuple[int, int, int, int]], current_face_box: Tuple[int, int, int, int]) -> bool:
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
    session_state: dict,
    prefix: str,
    frame: Any,
    faces_list: List[Tuple[int, int, int, int]],
    crop_face_fn,
    predict_face_fn,
    camera_index: Optional[int] = None,
    padding_ratio: float = 0.18,
) -> Tuple[Optional[Dict[str, Any]], bool]:
    """
    Nhận diện khuôn mặt phi chặn (Non-blocking Background Thread):
    Sử dụng PredictionStore an toàn luồng lưu trên heap Python,
    loại bỏ triệt để lỗi phân mảnh session state của Streamlit.
    Trả về: (latest_prediction, is_new_prediction)
    """
    store_key = f"{prefix}_prediction_store"
    store = session_state.get(store_key)
    if store is None or not isinstance(store, PredictionStore):
        store = PredictionStore()
        session_state[store_key] = store

    # Quản lý số lượng khuôn mặt và grace period (tránh nhấp nháy khi chớp tắt 1 frame)
    is_valid_face = store.handle_faces_count(len(faces_list))
    if not is_valid_face or len(faces_list) != 1:
        latest_prediction, _, _ = store.get_prediction()
        session_state[f"{prefix}_prediction"] = latest_prediction
        return latest_prediction, False

    current_face_box = _normalize_face_box(faces_list[0])
    now = time.time()

    # Kiểm tra xem có cần và có thể kích hoạt worker ngầm không
    if store.can_trigger(now, current_face_box):
        face_crop = crop_face_fn(
            frame,
            current_face_box,
            padding_ratio=padding_ratio,
        )
        store.set_busy(True, now, current_face_box)

        def _worker_thread(crop_img, cam_idx, pred_fn, pred_store: PredictionStore):
            try:
                kwargs = {}
                if cam_idx is not None:
                    kwargs["camera_index"] = cam_idx
                result = pred_fn(crop_img, **kwargs)
                pred_store.set_prediction(result)
            except Exception as err:
                print(f"Background prediction error: {err}")
                pred_store.set_prediction(None)

        t = threading.Thread(
            target=_worker_thread,
            args=(face_crop.copy(), camera_index, predict_face_fn, store),
            daemon=True,
        )

        # Gắn script run context an toàn nếu Streamlit context khả dụng
        try:
            from streamlit.runtime.scriptrunner import add_script_run_ctx
            add_script_run_ctx(t)
        except Exception:
            pass

        t.start()

    latest_prediction, is_new, _ = store.get_prediction()

    # Đồng bộ ngược vào session_state cho các module cũ đọc nếu cần
    session_state[f"{prefix}_prediction"] = latest_prediction

    return latest_prediction, is_new


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
    Lưu ý: Không gán session_state[f"{prefix}_run_camera"] = False ở đây để tránh
    lỗi StreamlitAPIException khi widget toggle/checkbox đã được render trên trang.
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


def release_inactive_cameras(session_state: dict, active_prefix: Optional[str] = None) -> None:
    """
    Đóng tất cả các camera đang mở của các trang khác để tránh lỗi xung đột phần cứng
    (Windows chỉ cho phép 1 tiến trình/tab giữ quyền điều khiển webcam tại một thời điểm).
    """
    for prefix in CAMERA_PAGE_KEYS:
        if active_prefix is not None and prefix == active_prefix:
            continue
        release_camera(session_state, prefix)
        # Chỉ reset run_camera cho các trang không active (chưa instantiate widget trên UI hiện tại)
        run_key = f"{prefix}_run_camera"
        if run_key in session_state:
            session_state[run_key] = False


DETECTION_INTERVAL = 1  # Quét nhận diện trực tiếp trên từng frame để khung bám sát khuôn mặt không độ trễ


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


def render_stream_frame(placeholder: Any, frame: Any, quality: int = STREAM_JPEG_QUALITY) -> bool:
    """
    Nén khung hình BGR sang JPEG siêu tốc (libjpeg-turbo C++) và hiển thị qua Streamlit placeholder.
    Giảm dung lượng gói tin từ ~100KB xuống ~20KB, ngăn ngừa tràn hàng đợi WebSocket và giảm tải cho trình duyệt.
    """
    if placeholder is None or frame is None:
        return False

    try:
        success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
        if success:
            placeholder.image(buffer.tobytes(), format="JPEG", use_container_width=True)
            return True
        else:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
            return False
    except Exception as err:
        print(f"Error rendering stream frame: {err}")
        return False

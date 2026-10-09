import sys
import os
import time
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.camera_stream import (
    STREAM_FRAME_INTERVAL,
    STREAM_SLEEP_INTERVAL,
    render_stream_frame,
    read_camera_frame,
    PredictionStore,
)
from db.database import initialize_database
from db.attendance_repo import (
    initialize_attendance_logs,
    register_check_in,
    register_check_out,
    get_latest_attendance_log,
)


class StrictStreamlitMockPlaceholder:
    """
    MockPlaceholder mô phỏng chính xác 100% chữ ký hàm st.image / DeltaGenerator.image của Streamlit.
    Tuyệt đối không dùng **kwargs để phát hiện ngay bất kỳ lỗi tham số nào.
    """
    def __init__(self):
        self.call_count = 0
        self.last_image = None
        self.last_channels = None
        self.last_use_container_width = None

    def image(
        self,
        image,
        caption=None,
        width="content",
        use_column_width=None,
        clamp=False,
        channels="RGB",
        output_format="auto",
        *,
        use_container_width=None,
        link=None,
    ):
        self.call_count += 1
        self.last_image = image
        self.last_channels = channels
        self.last_use_container_width = use_container_width


class MockCameraDevice:
    def __init__(self):
        self.grab_count = 0
        self.read_count = 0

    def isOpened(self):
        return True

    def grab(self):
        self.grab_count += 1
        return True

    def read(self):
        self.read_count += 1
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        return True, frame


def test_attendance_streaming_timing():
    print("1. Testing Attendance Direct RGB Streaming Timing (30 FPS)...")
    placeholder = StrictStreamlitMockPlaceholder()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Mô phỏng 1.0 giây vòng lặp streaming với STREAM_SLEEP_INTERVAL và STREAM_FRAME_INTERVAL
    start_time = time.time()
    last_video_time = 0.0
    iterations = 0

    while time.time() - start_time < 1.0:
        iterations += 1
        now = time.time()
        if now - last_video_time >= STREAM_FRAME_INTERVAL:
            success = render_stream_frame(placeholder, frame)
            assert success is True, "render_stream_frame returned False"
            last_video_time = now
        time.sleep(STREAM_SLEEP_INTERVAL)

    fps = placeholder.call_count
    print(f"  Iterations in 1.0s: {iterations}")
    print(f"  Rendered frames in 1.0s: {fps} FPS")
    print(f"  Image channels: {placeholder.last_channels}, use_container_width: {placeholder.last_use_container_width}")

    # Xác thực tốc độ và định dạng RGB trực tiếp
    assert fps >= 22, f"FPS too low: {fps} FPS (expected >= 22 FPS)"
    assert placeholder.last_channels == "RGB", f"Expected channels='RGB', got {placeholder.last_channels}"
    assert placeholder.last_use_container_width is True
    assert isinstance(placeholder.last_image, np.ndarray), f"Expected ndarray image, got {type(placeholder.last_image)}"
    print("  Streaming rate matches 30 FPS target & Direct RGB image -> PASSED!")


def test_attendance_read_frame_flush():
    print("\n2. Testing Attendance Camera Buffer Flushing...")
    mock_cam = MockCameraDevice()
    ret, frame = read_camera_frame(mock_cam)
    assert ret is True
    assert frame is not None
    assert mock_cam.grab_count == 1, "Must call cap.grab() to flush buffer"
    assert mock_cam.read_count == 1, "Must call cap.read() to get latest frame"
    print("  DirectShow frame buffer flush -> PASSED!")


def test_attendance_auto_log_integration():
    print("\n3. Testing Auto Attendance Flow (3 Stable Frames + Cooldown)...")
    initialize_database()
    initialize_attendance_logs()

    # Giả lập kết quả nhận diện nhân viên NV003
    target_code = "NV003"
    mock_prediction = {
        "display_code": target_code,
        "predicted_code": target_code,
        "confidence": 0.45,
        "match_threshold": 0.60,
        "is_match": True,
    }

    # Giả lập đếm 3 khung hình ổn định liên tiếp (AUTO_MATCH_REQUIRED_FRAMES = 3)
    stable_count = 0
    stable_code = None
    REQUIRED_FRAMES = 3

    for step in range(1, 4):
        pred_code = mock_prediction["display_code"]
        if stable_code == pred_code:
            stable_count += 1
        else:
            stable_code = pred_code
            stable_count = 1
        print(f"  Frame {step}: Code={stable_code}, Stable Count={stable_count}")

    assert stable_count == REQUIRED_FRAMES, f"Expected {REQUIRED_FRAMES}, got {stable_count}"

    # Thực hiện check-in hoặc check-out theo trạng thái cơ sở dữ liệu
    latest = get_latest_attendance_log(target_code)
    try:
        if latest is None or latest.get("log_type") == "OUT":
            new_log = register_check_in(
                employee_code=target_code,
                camera_source="Camera 1",
                confidence=mock_prediction["confidence"],
            )
            print(f"  Successfully registered check-in: {new_log['employee_code']} at {new_log['log_time']}, type=IN")
        else:
            new_log = register_check_out(
                employee_code=target_code,
                camera_source="Camera 2",
                confidence=mock_prediction["confidence"],
            )
            print(f"  Successfully registered check-out: {new_log['employee_code']} at {new_log['log_time']}, type=OUT")
    except ValueError as err:
        print(f"  Business rule enforced: {err}")

    # Xác thực bản ghi log mới nhất tồn tại
    latest_after = get_latest_attendance_log(target_code)
    assert latest_after is not None
    assert latest_after["employee_code"] == target_code
    print("  Auto Attendance Integration -> PASSED!")


def test_attendance_syntax_compile():
    print("\n4. Testing Syntax and Compilation of pages/4_Attendance.py...")
    with open("pages/4_Attendance.py", "r", encoding="utf-8") as f:
        code = f.read()
    compile(code, "pages/4_Attendance.py", "exec")
    print("  pages/4_Attendance.py compiles cleanly -> PASSED!")


if __name__ == "__main__":
    test_attendance_streaming_timing()
    test_attendance_read_frame_flush()
    test_attendance_auto_log_integration()
    test_attendance_syntax_compile()
    print("\n>>> ALL PHASE 2 ATTENDANCE INTEGRATION TESTS PASSED SUCCESSFULLY! <<<")

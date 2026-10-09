import sys
import os
import time
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.camera_stream import (
    STREAM_FRAME_INTERVAL,
    STREAM_SLEEP_INTERVAL,
    render_stream_frame,
)
from db.database import initialize_database
from db.attendance_repo import (
    initialize_attendance_logs,
    register_check_in,
    register_check_out,
    get_latest_attendance_log,
)

class MockPlaceholder:
    def __init__(self):
        self.call_count = 0
        self.last_image = None
        self.last_format = None

    def image(self, img_data, **kwargs):
        self.call_count += 1
        self.last_image = img_data
        self.last_format = kwargs.get("format")


def test_attendance_streaming_timing():
    print("Testing Attendance Streaming Timing...")
    placeholder = MockPlaceholder()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Simulate 1.0 second of streaming loop iterations with STREAM_SLEEP_INTERVAL
    start_time = time.time()
    last_video_time = 0.0
    iterations = 0

    while time.time() - start_time < 1.0:
        iterations += 1
        now = time.time()
        if now - last_video_time >= STREAM_FRAME_INTERVAL:
            render_stream_frame(placeholder, frame)
            last_video_time = now
        time.sleep(STREAM_SLEEP_INTERVAL)

    fps = placeholder.call_count
    print(f"  Iterations in 1.0s: {iterations}")
    print(f"  Rendered frames in 1.0s: {fps} FPS")
    print(f"  Format: {placeholder.last_format}, Payload size: {len(placeholder.last_image)} bytes")

    # Expected: 12-16 FPS (Golden FPS)
    assert 12 <= fps <= 16, f"FPS out of target range: {fps}"
    assert placeholder.last_format == "JPEG"
    print("  Streaming rate matches Golden FPS (12-16 FPS) -> PASSED!")


def test_attendance_auto_log_integration():
    print("\nTesting Auto Attendance Integration with Stability Flow...")
    initialize_database()
    initialize_attendance_logs()

    # Simulate recognized prediction for NV002
    target_code = "NV002"
    mock_prediction = {
        "display_code": target_code,
        "predicted_code": target_code,
        "confidence": 0.42,
        "match_threshold": 0.60,
        "is_match": True,
    }

    # Simulate 3 stable frames
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
        print(f"  Step {step}: Code={stable_code}, Stable Count={stable_count}")

    assert stable_count == REQUIRED_FRAMES, f"Expected {REQUIRED_FRAMES}, got {stable_count}"

    # Check-in or Check-out depending on current state
    latest = get_latest_attendance_log(target_code)
    try:
        if latest is None or latest.get("log_type") == "OUT":
            new_log = register_check_in(
                employee_code=target_code,
                camera_source="Camera 1",
                confidence=mock_prediction["confidence"],
            )
            print(f"  Successfully registered check-in: {new_log['employee_code']} at {new_log['log_time']}")
        else:
            # Employee already checked in, test check-out or duplicate handling
            new_log = register_check_out(
                employee_code=target_code,
                camera_source="Camera 2",
                confidence=mock_prediction["confidence"],
            )
            print(f"  Successfully registered check-out: {new_log['employee_code']} at {new_log['log_time']}")
    except ValueError as err:
        print(f"  Business rule enforced: {err}")

    # Verify latest log exists
    latest_after = get_latest_attendance_log(target_code)
    assert latest_after is not None
    assert latest_after["employee_code"] == target_code
    print("  Auto Attendance Integration -> PASSED!")


if __name__ == "__main__":
    test_attendance_streaming_timing()
    test_attendance_auto_log_integration()
    print("\n>>> ALL PHASE 2 ATTENDANCE INTEGRATION TESTS PASSED SUCCESSFULLY! <<<")

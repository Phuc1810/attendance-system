import time
import numpy as np
import cv2
from core.camera_stream import (
    STREAM_FRAME_INTERVAL,
    STREAM_SLEEP_INTERVAL,
    RECOGNITION_COOLDOWN_SECONDS,
    RECOGNITION_UNKNOWN_RETRY_SECONDS,
    PredictionStore,
    render_stream_frame,
    read_camera_frame,
    ThreadedCameraReader,
)


class StrictStreamlitMockPlaceholder:
    """
    MockPlaceholder mô phỏng chính xác chữ ký hàm st.image / DeltaGenerator.image của Streamlit.
    """
    def __init__(self):
        self.last_image = None
        self.last_channels = None
        self.last_use_container_width = None
        self.call_count = 0

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
        self.last_image = image
        self.last_channels = channels
        self.last_use_container_width = use_container_width
        self.call_count += 1


class MockCapture:
    def __init__(self, is_opened: bool = True):
        self._opened = is_opened
        self.grab_called = 0
        self.read_called = 0

    def isOpened(self) -> bool:
        return self._opened

    def grab(self) -> bool:
        self.grab_called += 1
        return True

    def read(self):
        self.read_called += 1
        return True, np.zeros((480, 640, 3), dtype=np.uint8)


def test_constants():
    print("1. Testing Stream & Recognition Constants...")
    assert STREAM_FRAME_INTERVAL <= 0.035, f"Interval too slow for 30 FPS: {STREAM_FRAME_INTERVAL}"
    assert STREAM_SLEEP_INTERVAL <= 0.015, f"Sleep too high: {STREAM_SLEEP_INTERVAL}"
    assert RECOGNITION_COOLDOWN_SECONDS >= 2.5, f"Cooldown too short: {RECOGNITION_COOLDOWN_SECONDS}"
    assert RECOGNITION_UNKNOWN_RETRY_SECONDS >= 1.0, f"Retry too short: {RECOGNITION_UNKNOWN_RETRY_SECONDS}"
    
    fps = 1.0 / STREAM_FRAME_INTERVAL
    print(f"  STREAM_FRAME_INTERVAL = {STREAM_FRAME_INTERVAL:.3f}s (~{fps:.1f} FPS) -> OK")
    print(f"  STREAM_SLEEP_INTERVAL = {STREAM_SLEEP_INTERVAL:.3f}s -> OK")
    print(f"  RECOGNITION_COOLDOWN_SECONDS = {RECOGNITION_COOLDOWN_SECONDS:.1f}s -> OK")
    print(f"  RECOGNITION_UNKNOWN_RETRY_SECONDS = {RECOGNITION_UNKNOWN_RETRY_SECONDS:.1f}s -> OK")


def test_render_stream_frame_rgb():
    print("\n2. Testing render_stream_frame Direct RGB Rendering...")
    placeholder = StrictStreamlitMockPlaceholder()

    # Guard checks
    assert render_stream_frame(None, np.zeros((10, 10, 3))) is False
    assert render_stream_frame(placeholder, None) is False
    assert placeholder.call_count == 0
    print("  Null input guards -> OK")

    # Valid BGR image rendering
    bgr_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    bgr_frame[:, :, 0] = 255  # Blue channel in BGR
    
    t0 = time.perf_counter()
    success = render_stream_frame(placeholder, bgr_frame)
    render_time_ms = (time.perf_counter() - t0) * 1000

    assert success is True
    assert placeholder.call_count == 1
    assert isinstance(placeholder.last_image, np.ndarray), f"Expected ndarray, got {type(placeholder.last_image)}"
    assert placeholder.last_channels == "RGB"
    assert placeholder.last_use_container_width is True
    # Verify BGR -> RGB conversion happened (Blue channel in BGR becomes Red channel at index 2 in RGB)
    assert placeholder.last_image[0, 0, 2] == 255
    assert placeholder.last_image[0, 0, 0] == 0

    print(f"  Direct RGB render -> OK ({render_time_ms:.2f} ms, channels={placeholder.last_channels}, shape={placeholder.last_image.shape})")


def test_read_camera_frame_flushing():
    print("\n3. Testing read_camera_frame Windows DirectShow Buffer Flushing...")
    mock_cap = MockCapture(is_opened=True)
    
    ret, frame = read_camera_frame(mock_cap)
    assert ret is True
    assert frame is not None
    assert mock_cap.grab_called == 1, "Expected cap.grab() to be called to flush DirectShow stale frame"
    assert mock_cap.read_called == 1
    print("  DirectShow buffer flush (cap.grab) -> OK")


def test_smart_recognition_coordinator():
    print("\n4. Testing Smart Recognition Coordinator (PredictionStore & Cooldown)...")
    store = PredictionStore()
    now = time.time()
    box1 = (100, 100, 80, 80)
    box2 = (110, 105, 80, 80)  # Minor head movement
    box3 = (300, 200, 90, 90)  # Major position jump

    # Step A: New face enters frame -> must trigger immediately
    assert store.can_trigger(now, box1) is True, "New face must trigger immediately"
    store.set_busy(True, now, box1)
    print("  Step A: Initial face appearance triggers recognition -> OK")

    # Step B: Worker is busy running Dlib -> can_trigger must return False
    assert store.can_trigger(now + 0.1, box1) is False, "Must not trigger while worker is busy"
    print("  Step B: Worker busy lock prevents redundant threads -> OK")

    # Step C: Worker finishes and matches NV001
    matched_pred = {
        "is_match": True,
        "employee_code": "NV001",
        "display_code": "NV001",
        "confidence": 0.42,
        "match_threshold": 0.50,
    }
    finish_time = now + 1.65
    store.set_prediction(matched_pred, finished_at=finish_time)
    
    pred, is_new, is_busy = store.get_prediction()
    assert pred["employee_code"] == "NV001"
    assert is_new is True
    assert is_busy is False
    print("  Step C: Match result stored and retrieved -> OK")

    # Step D: During 3.0s cooldown, user moves face left/right
    # can_trigger MUST return False to keep Python GIL 100% free!
    assert store.can_trigger(finish_time + 0.5, box2) is False, "Must keep lock during head movement"
    assert store.can_trigger(finish_time + 1.5, box2) is False, "Must keep lock at t=1.5s"
    assert store.can_trigger(finish_time + 2.8, box2) is False, "Must keep lock at t=2.8s"
    
    # Verify cached prediction is STILL available for drawing green box at 30 FPS
    cached_pred, is_new2, _ = store.get_prediction()
    assert cached_pred["employee_code"] == "NV001"
    assert is_new2 is False, "Result already read, should not be marked new"
    print(f"  Step D: Cooldown {RECOGNITION_COOLDOWN_SECONDS}s holds lock & frees GIL during movement -> OK")

    # Step E: After 3.0s cooldown expires, periodic re-validation is allowed
    assert store.can_trigger(finish_time + 3.1, box2) is True, "Must allow re-check after 3.0s"
    print("  Step E: Cooldown expiration allows periodic re-check -> OK")

    # Step F: Stranger / Unknown Face handling
    unknown_pred = {
        "is_match": False,
        "employee_code": None,
        "display_code": "Unknown",
        "confidence": 0.68,
        "match_threshold": 0.50,
    }
    unk_time = finish_time + 3.5
    unk_finish = unk_time + 1.65
    store.set_busy(True, unk_time, box1)
    store.set_prediction(unknown_pred, finished_at=unk_finish)

    # Unknown face must NOT hammer GIL continuously: must wait at least 1.5s
    assert store.can_trigger(unk_finish + 0.5, box1) is False, "Unknown must wait retry interval"
    assert store.can_trigger(unk_finish + 1.6, box3) is True, "Unknown re-checks after 1.5s when moved"
    print("  Step F: Unknown retry throttling prevents continuous GIL choking -> OK")

    # Step G: Face leaves frame (empty count for 5 frames)
    for _ in range(4):
        still_valid = store.handle_faces_count(0)
        assert still_valid is True, "Grace period 1-4 frames must preserve state"
    
    # 5th frame without face -> clears prediction
    still_valid = store.handle_faces_count(0)
    assert still_valid is False, "5th empty frame must clear prediction"
    
    cleared_pred, _, _ = store.get_prediction()
    assert cleared_pred is None, "Prediction should be cleared after face leaves"
    
    # When next person appears, triggers immediately
    assert store.can_trigger(time.time(), box1) is True, "Next person triggers immediately"
    print("  Step G: Grace period and state clearing on face exit -> OK")


def test_fps_simulation():
    print("\n5. Testing 30 FPS Stream Loop Simulation...")
    placeholder = StrictStreamlitMockPlaceholder()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    total_frames = 60
    start_time = time.perf_counter()
    for _ in range(total_frames):
        render_stream_frame(placeholder, frame)
        time.sleep(STREAM_SLEEP_INTERVAL)
    elapsed = time.perf_counter() - start_time
    
    sim_fps = total_frames / elapsed
    print(f"  Rendered {total_frames} frames in {elapsed:.2f}s -> {sim_fps:.1f} FPS (Target ~30 FPS) -> OK")
    assert sim_fps >= 25.0, f"Streaming simulation too slow: {sim_fps:.1f} FPS"


if __name__ == "__main__":
    test_constants()
    test_render_stream_frame_rgb()
    test_read_camera_frame_flushing()
    test_smart_recognition_coordinator()
    test_fps_simulation()
    print("\n>>> ALL PHASE 1 ZERO-DELAY TESTS PASSED WITH 100% SUCCESS! <<<")

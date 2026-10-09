import sys
import os
import time
import numpy as np
import cv2

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.camera_stream import (
    STREAM_FRAME_INTERVAL,
    STREAM_SLEEP_INTERVAL,
    render_stream_frame,
    PredictionStore,
)
from db.database import initialize_database


class StrictStreamlitMockPlaceholder:
    """
    MockPlaceholder mô phỏng chính xác 100% chữ ký hàm st.image / DeltaGenerator.image của Streamlit.
    Tuyệt đối không dùng **kwargs để phát hiện ngay bất kỳ lỗi tham số nào.
    """
    def __init__(self, simulate_bytes_error: bool = False):
        self.call_count = 0
        self.last_image = None
        self.last_output_format = None
        self.last_channels = None
        self.simulate_bytes_error = simulate_bytes_error

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
        if self.simulate_bytes_error and isinstance(image, bytes):
            raise RuntimeError("Simulated Streamlit bytes error to trigger Tier 2 fallback")

        self.call_count += 1
        self.last_image = image
        self.last_output_format = output_format
        self.last_channels = channels


def test_face_recognition_streaming_timing():
    print("Testing Face Recognition Streaming Timing with Strict Streamlit Mock...")
    placeholder = StrictStreamlitMockPlaceholder()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Mô phỏng 1.0 giây streaming loop với STREAM_SLEEP_INTERVAL và STREAM_FRAME_INTERVAL
    start_time = time.time()
    last_video_time = 0.0
    iterations = 0

    while time.time() - start_time < 1.0:
        iterations += 1
        now = time.time()
        if now - last_video_time >= STREAM_FRAME_INTERVAL:
            success = render_stream_frame(placeholder, frame)
            assert success is True, "render_stream_frame failed"
            last_video_time = now
        time.sleep(STREAM_SLEEP_INTERVAL)

    fps = placeholder.call_count
    print(f"  Iterations in 1.0s: {iterations}")
    print(f"  Rendered frames in 1.0s: {fps} FPS")
    print(f"  Output Format: {placeholder.last_output_format}, Payload size: {len(placeholder.last_image)} bytes")

    # Xác thực tốc độ và định dạng
    assert 12 <= fps <= 16, f"FPS out of target range: {fps}"
    assert placeholder.last_output_format == "JPEG", f"Expected output_format='JPEG', got {placeholder.last_output_format}"
    assert isinstance(placeholder.last_image, bytes)
    print("  Streaming rate matches Golden FPS (12-16 FPS) & output_format='JPEG' -> PASSED!")


def test_face_recognition_annotation_and_render():
    print("\nTesting Face Recognition Bounding Box Annotation and JPEG Rendering...")
    placeholder = StrictStreamlitMockPlaceholder()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # 1. Test Match Annotation (Xanh lá)
    faces_list = [(100, 100, 200, 200)]
    pred_match = {
        "display_code": "NV001",
        "predicted_code": "NV001",
        "confidence": 0.45,
        "match_threshold": 0.60,
        "is_match": True,
    }

    annotated = frame.copy()
    for (x, y, w, h) in faces_list:
        label = f"{pred_match['display_code']} ({pred_match['confidence']:.2f})"
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(annotated, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    t0 = time.perf_counter()
    success = render_stream_frame(placeholder, annotated)
    encode_ms = (time.perf_counter() - t0) * 1000

    assert success is True
    assert placeholder.last_output_format == "JPEG"
    assert isinstance(placeholder.last_image, bytes)
    print(f"  Rendered Match Frame: {len(placeholder.last_image)} bytes in {encode_ms:.2f}ms (output_format=JPEG) -> PASSED!")

    # 2. Test Unknown Annotation (Đỏ)
    pred_unknown = {
        "display_code": "Unknown",
        "predicted_code": "Unknown",
        "confidence": 0.75,
        "match_threshold": 0.60,
        "is_match": False,
    }
    annotated_unk = frame.copy()
    for (x, y, w, h) in faces_list:
        label = f"Unknown ({pred_unknown['confidence']:.2f})"
        cv2.rectangle(annotated_unk, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(annotated_unk, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    success_unk = render_stream_frame(placeholder, annotated_unk)
    assert success_unk is True
    assert placeholder.last_output_format == "JPEG"
    print(f"  Rendered Unknown Frame: {len(placeholder.last_image)} bytes -> PASSED!")


def test_face_recognition_fallback_tier():
    print("\nTesting Face Recognition Tier 2 Fallback Execution...")
    fallback_placeholder = StrictStreamlitMockPlaceholder(simulate_bytes_error=True)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    success = render_stream_frame(fallback_placeholder, frame)
    assert success is True, "Fallback render failed"
    assert fallback_placeholder.call_count == 1
    assert isinstance(fallback_placeholder.last_image, np.ndarray)
    assert fallback_placeholder.last_channels == "RGB"
    print(f"  Tier 2 Fallback successfully activated: channels={fallback_placeholder.last_channels} -> PASSED!")


def test_both_pages_syntax_and_import():
    print("\nTesting Syntax and Compilation Integrity of Both Target Pages...")
    for page_path in ["pages/4_Attendance.py", "pages/7_Face_Recognition.py"]:
        with open(page_path, "r", encoding="utf-8") as f:
            code = f.read()
        compile(code, page_path, "exec")
        print(f"  Compiled successfully: {page_path} -> OK")


if __name__ == "__main__":
    test_face_recognition_streaming_timing()
    test_face_recognition_annotation_and_render()
    test_face_recognition_fallback_tier()
    test_both_pages_syntax_and_import()
    print("\n>>> ALL PHASE 3 FACE RECOGNITION INTEGRATION TESTS PASSED SUCCESSFULLY! <<<")

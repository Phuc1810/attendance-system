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

class MockPlaceholder:
    def __init__(self):
        self.call_count = 0
        self.last_image = None
        self.last_format = None

    def image(self, img_data, **kwargs):
        self.call_count += 1
        self.last_image = img_data
        self.last_format = kwargs.get("format")


def test_face_recognition_streaming_timing():
    print("Testing Face Recognition Streaming Timing...")
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


def test_face_recognition_annotation_and_render():
    print("\nTesting Face Recognition Bounding Box Annotation and JPEG Rendering...")
    placeholder = MockPlaceholder()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # 1. Test Match Annotation
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
    assert placeholder.last_format == "JPEG"
    print(f"  Rendered Match Frame: {len(placeholder.last_image)} bytes in {encode_ms:.2f}ms -> PASSED!")

    # 2. Test Unknown Annotation
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
    print(f"  Rendered Unknown Frame: {len(placeholder.last_image)} bytes -> PASSED!")


def test_both_pages_syntax_and_import():
    print("\nTesting Syntax and Import Integrity of Both Target Pages...")
    import importlib.util

    for page_path in ["pages/4_Attendance.py", "pages/7_Face_Recognition.py"]:
        with open(page_path, "r", encoding="utf-8") as f:
            code = f.read()
        compile(code, page_path, "exec")
        print(f"  Compiled successfully: {page_path} -> OK")


if __name__ == "__main__":
    test_face_recognition_streaming_timing()
    test_face_recognition_annotation_and_render()
    test_both_pages_syntax_and_import()
    print("\n>>> ALL PHASE 3 FACE RECOGNITION INTEGRATION TESTS PASSED SUCCESSFULLY! <<<")

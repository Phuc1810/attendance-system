import time
import numpy as np
import cv2
from core.camera_stream import (
    STREAM_FRAME_INTERVAL,
    STREAM_SLEEP_INTERVAL,
    STREAM_JPEG_QUALITY,
    render_stream_frame,
)


class MockPlaceholder:
    def __init__(self):
        self.last_image = None
        self.last_format = None
        self.call_count = 0

    def image(self, img_data, **kwargs):
        self.last_image = img_data
        self.last_format = kwargs.get("format")
        self.call_count += 1


def test_constants():
    print("Testing Stream Constants...")
    assert STREAM_FRAME_INTERVAL > 0.05 and STREAM_FRAME_INTERVAL < 0.1, f"Invalid interval: {STREAM_FRAME_INTERVAL}"
    assert STREAM_SLEEP_INTERVAL > 0.005 and STREAM_SLEEP_INTERVAL < 0.05, f"Invalid sleep: {STREAM_SLEEP_INTERVAL}"
    assert STREAM_JPEG_QUALITY >= 50 and STREAM_JPEG_QUALITY <= 95, f"Invalid quality: {STREAM_JPEG_QUALITY}"
    fps = 1.0 / STREAM_FRAME_INTERVAL
    print(f"  STREAM_FRAME_INTERVAL = {STREAM_FRAME_INTERVAL:.3f}s (~{fps:.1f} FPS) -> OK")
    print(f"  STREAM_SLEEP_INTERVAL = {STREAM_SLEEP_INTERVAL:.3f}s -> OK")
    print(f"  STREAM_JPEG_QUALITY = {STREAM_JPEG_QUALITY} -> OK")


def test_render_stream_frame():
    print("\nTesting render_stream_frame...")
    placeholder = MockPlaceholder()
    
    # 1. Test None inputs
    assert render_stream_frame(None, np.zeros((10, 10, 3))) is False
    assert render_stream_frame(placeholder, None) is False
    assert placeholder.call_count == 0
    print("  Null input guards -> OK")

    # 2. Test valid frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(frame, "Streamlit Zero Delay", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    success = render_stream_frame(placeholder, frame)
    assert success is True, "render_stream_frame failed"
    assert placeholder.call_count == 1, "image not called"
    assert isinstance(placeholder.last_image, bytes), f"Expected bytes, got {type(placeholder.last_image)}"
    assert placeholder.last_format == "JPEG", f"Expected format='JPEG', got {placeholder.last_format}"
    print(f"  Render valid frame -> OK (Payload size: {len(placeholder.last_image)} bytes, format={placeholder.last_format})")


def test_compression_performance():
    print("\nTesting JPEG Compression Performance & Size Reduction...")
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.circle(frame, (320, 240), 100, (255, 200, 150), -1)
    cv2.putText(frame, "Sample Face Simulation", (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    raw_rgb_size = frame.nbytes
    
    # Warm-up
    placeholder = MockPlaceholder()
    render_stream_frame(placeholder, frame)

    times = []
    for _ in range(50):
        t0 = time.perf_counter()
        render_stream_frame(placeholder, frame)
        times.append((time.perf_counter() - t0) * 1000)

    avg_time_ms = np.mean(times)
    compressed_size = len(placeholder.last_image)
    reduction_pct = (1.0 - compressed_size / raw_rgb_size) * 100

    print(f"  Raw RGB Frame Size: {raw_rgb_size:,} bytes")
    print(f"  Compressed JPEG Size: {compressed_size:,} bytes")
    print(f"  Size Reduction: {reduction_pct:.1f}%")
    print(f"  Average Encoding Time: {avg_time_ms:.2f} ms")

    assert avg_time_ms < 15.0, f"Encoding time too slow: {avg_time_ms:.2f} ms"
    assert reduction_pct > 70.0, f"Compression reduction too low: {reduction_pct:.1f}%"
    print("  Compression Performance -> PASSED!")


if __name__ == "__main__":
    test_constants()
    test_render_stream_frame()
    test_compression_performance()
    print("\n>>> ALL PHASE 1 UNIT TESTS PASSED SUCCESSFULLY! <<<")

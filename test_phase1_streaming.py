import time
import numpy as np
import cv2
from core.camera_stream import (
    STREAM_FRAME_INTERVAL,
    STREAM_SLEEP_INTERVAL,
    STREAM_JPEG_QUALITY,
    render_stream_frame,
)


class StrictStreamlitMockPlaceholder:
    """
    MockPlaceholder mô phỏng chính xác 100% chữ ký hàm st.image / DeltaGenerator.image của Streamlit.
    Tuyệt đối không dùng **kwargs để phát hiện ngay bất kỳ tham số sai tên nào (ví dụ format vs output_format).
    """
    def __init__(self, simulate_bytes_error: bool = False):
        self.last_image = None
        self.last_output_format = None
        self.last_channels = None
        self.call_count = 0
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
            raise RuntimeError("Simulated Streamlit bytes transfer error to trigger Tier 2 fallback")

        self.last_image = image
        self.last_output_format = output_format
        self.last_channels = channels
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


def test_render_stream_frame_tier1_jpeg():
    print("\nTesting render_stream_frame Tier 1 (Turbo JPEG with output_format='JPEG')...")
    placeholder = StrictStreamlitMockPlaceholder()

    # 1. Test None inputs
    assert render_stream_frame(None, np.zeros((10, 10, 3))) is False
    assert render_stream_frame(placeholder, None) is False
    assert placeholder.call_count == 0
    print("  Null input guards -> OK")

    # 2. Test valid frame with Turbo JPEG
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(frame, "Streamlit Zero Delay", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    success = render_stream_frame(placeholder, frame)
    assert success is True, "render_stream_frame failed"
    assert placeholder.call_count == 1, "image not called"
    assert isinstance(placeholder.last_image, bytes), f"Expected bytes, got {type(placeholder.last_image)}"
    assert placeholder.last_output_format == "JPEG", f"Expected output_format='JPEG', got {placeholder.last_output_format}"
    print(f"  Tier 1 JPEG render -> OK (Payload size: {len(placeholder.last_image)} bytes, output_format={placeholder.last_output_format})")


def test_render_stream_frame_tier2_fallback():
    print("\nTesting render_stream_frame Tier 2 (Automatic RGB Fallback)...")
    # Khởi tạo placeholder với cờ simulate_bytes_error để ép lỗi ở Tầng 1
    placeholder = StrictStreamlitMockPlaceholder(simulate_bytes_error=True)

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.circle(frame, (100, 100), 40, (0, 0, 255), -1)

    # Khi Tầng 1 ném lỗi, hàm phải tự động chuyển sang Tầng 2 và render thành công
    success = render_stream_frame(placeholder, frame)
    assert success is True, "Fallback Tier 2 failed"
    assert placeholder.call_count == 1, "Fallback image not called"
    assert isinstance(placeholder.last_image, np.ndarray), f"Expected ndarray for fallback, got {type(placeholder.last_image)}"
    assert placeholder.last_channels == "RGB", f"Expected channels='RGB', got {placeholder.last_channels}"
    print(f"  Tier 2 Fallback render -> OK (Successfully rendered RGB array, shape={placeholder.last_image.shape}, channels={placeholder.last_channels})")


def test_compression_performance():
    print("\nTesting JPEG Compression Performance & Size Reduction...")
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.circle(frame, (320, 240), 100, (255, 200, 150), -1)
    cv2.putText(frame, "Sample Face Simulation", (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    raw_rgb_size = frame.nbytes

    placeholder = StrictStreamlitMockPlaceholder()
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
    test_render_stream_frame_tier1_jpeg()
    test_render_stream_frame_tier2_fallback()
    test_compression_performance()
    print("\n>>> ALL PHASE 1 UNIT & FALLBACK TESTS PASSED SUCCESSFULLY! <<<")

import cv2
import face_recognition

MAX_PROCESSING_WIDTH = 320
MIN_FACE_SIZE = (50, 50)
# Chuyển sang sử dụng face_recognition (dlib HOG) thay cho Haar Cascade
# HOG nhanh trên CPU, chính xác hơn Haar Cascade nhiều lần
# Tự động tính toán scale để đảm bảo tốc độ không bị treo khi luồng video có độ phân giải cao

def detect_faces(frame):
    """
    Detect faces using face_recognition library (dlib HOG model).
    Returns face locations in (x, y, w, h) format for backward compatibility
    with the existing UI code, plus a grayscale image.
    """
    height, width = frame.shape[:2]
    
    # Calculate dynamic scale to ensure the processing width is at most MAX_PROCESSING_WIDTH
    scale = MAX_PROCESSING_WIDTH / width if width > MAX_PROCESSING_WIDTH else 1.0
    
    resized_width = max(1, int(width * scale))
    resized_height = max(1, int(height * scale))

    # face_recognition expects RGB
    small_frame = cv2.resize(frame, (resized_width, resized_height))
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # face_locations returns list of (top, right, bottom, left) tuples
    face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")

    # Convert to (x, y, w, h) format and scale back to original size
    scaled_faces = []
    for (top, right, bottom, left) in face_locations:
        x = int(left / scale)
        y = int(top / scale)
        w = int((right - left) / scale)
        h = int((bottom - top) / scale)

        # Filter out faces smaller than minimum size
        if w >= MIN_FACE_SIZE[0] and h >= MIN_FACE_SIZE[1]:
            scaled_faces.append((x, y, w, h))

    gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
    return scaled_faces, gray

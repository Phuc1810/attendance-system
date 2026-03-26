import cv2

DETECTION_SCALE = 0.5
MIN_FACE_SIZE = (50, 50)

#Mục tiêu: nhận diện bằng OpenCV để tìm vị trí mặt
# Load model Haar Cascade co san trong OpenCV
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def detect_faces(frame):
    """
    Detect faces on a smaller frame to reduce CPU usage, then scale
    coordinates back to the original frame size.
    """
    height, width = frame.shape[:2]
    resized_width = max(1, int(width * DETECTION_SCALE))
    resized_height = max(1, int(height * DETECTION_SCALE))

    resized_frame = cv2.resize(frame, (resized_width, resized_height))
    gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)

    scaled_min_face = (
        max(20, int(MIN_FACE_SIZE[0] * DETECTION_SCALE)),
        max(20, int(MIN_FACE_SIZE[1] * DETECTION_SCALE)),
    )

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=scaled_min_face,
    )

    scaled_faces = [
        (
            int(x / DETECTION_SCALE),
            int(y / DETECTION_SCALE),
            int(w / DETECTION_SCALE),
            int(h / DETECTION_SCALE),
        )
        for (x, y, w, h) in faces
    ]

    return scaled_faces, gray

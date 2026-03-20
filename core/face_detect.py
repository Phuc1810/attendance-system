import cv2 #Nạp thư viện Opencv

#Load model Haar Cascade có sẵn trong OpenCV
#cv2.data.haarcascades: đây là đường dẫn thư mục chứa sẵn file Haar Cascade của OpenCV
# cv2.CascadeClassifier: dùng để load model phát hiện khuôn mặt
face_cascade = cv2.CascadeClassifier(
     cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# hàm nhận 1 ảnh từ camera và trả về danh sách khuôn mặt
def detect_faces(frame):
     
    #Nhận vào 1 frame ảnh màu từ camera
    #Trả về:
    #- faces: danh sách các khuôn mặt tìm được
    #- gray: ảnh xám để debug nếu cần

    #Chuyển ảnh màu sang ảnh xám
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    #Phát hiện có khuôn mặt trong ảnh
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor = 1.1,
        minNeighbors = 5,
        minSize = (50,50)
    )
    return faces, gray
import streamlit as st
import cv2
from core.face_detect import detect_faces

st.title("Face Detection")
CAM_INDEX = st.selectbox("Choose camera",[0,1])
run = st.checkbox("Run camera")

frame_placeholder = st.empty() # tạo vùng trống để cập nhật ảnh liên tục

if run:
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    while True:
        ret,frame = cap.read()
        if not ret:
            st.error("Cannot read frame from camera")
            break
        faces, gray = detect_faces(frame)

        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        
        cv2.putText(
            frame,
            f"Faces detected: {len(faces)}",
            (10,30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            2
        )

        frame_rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb,channels="RGB")

    cap.release()
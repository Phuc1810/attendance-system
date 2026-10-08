import cv2
import av
from streamlit_webrtc import VideoProcessorBase

from core.face_detect import detect_faces
from core.face_recognizer import predict_face
from core.save_face import crop_and_resize_face

class DetectionProcessor(VideoProcessorBase):
    def __init__(self):
        self.faces_list = []
        self.frame_bgr = None
        
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        self.frame_bgr = img.copy()
        
        faces_list, _ = detect_faces(img)
        self.faces_list = faces_list
        
        for (x, y, w, h) in faces_list:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
        cv2.putText(
            img,
            f"Faces detected: {len(faces_list)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        return av.VideoFrame.from_ndarray(img, format="bgr24")

class RecognitionProcessor(VideoProcessorBase):
    def __init__(self):
        self.faces_list = []
        self.frame_bgr = None
        self.prediction = None
        self.model_error = False
        self.camera_index = 0
        self.padding_ratio = 0.0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        self.frame_bgr = img.copy()
        
        faces_list, _ = detect_faces(img)
        self.faces_list = faces_list
        
        annotated_frame = img.copy()
        self.prediction = None
        self.model_error = False
        
        if len(faces_list) == 1:
            try:
                face_crop = crop_and_resize_face(
                    img,
                    faces_list[0],
                    padding_ratio=self.padding_ratio,
                )
                self.prediction = predict_face(face_crop, camera_index=self.camera_index)
            except Exception as e:
                print(f"Prediction error: {e}")
                self.model_error = True
                
        for (x, y, w, h) in faces_list:
            if len(faces_list) == 1 and self.prediction is not None:
                if self.prediction["is_match"]:
                    label_text = f"{self.prediction['display_code']} ({self.prediction['confidence']:.2f})"
                    box_color = (0, 255, 0)
                else:
                    label_text = f"Unknown ({self.prediction['confidence']:.2f} > {self.prediction['match_threshold']:.2f})"
                    box_color = (0, 0, 255)
            elif self.model_error:
                label_text = "Model error"
                box_color = (0, 0, 255)
            else:
                label_text = "Face detected"
                box_color = (0, 215, 255)
                
            cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), box_color, 2)
            cv2.putText(
                annotated_frame,
                label_text,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                box_color,
                2,
            )

        cv2.putText(
            annotated_frame,
            f"Faces detected: {len(faces_list)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

class AttendanceProcessor(VideoProcessorBase):
    def __init__(self):
        self.faces_list = []
        self.frame_bgr = None
        self.prediction = None
        self.model_error = False
        self.camera_index = 0
        self.padding_ratio = 0.0
        
        self.stable_employee_code = None
        self.stable_frame_count = 0
        
        self.recognized_employee_code = None
        self.recognized_confidence = None
        self.recognized_threshold = None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        self.frame_bgr = img.copy()
        
        faces_list, _ = detect_faces(img)
        self.faces_list = faces_list
        
        annotated_frame = img.copy()
        self.prediction = None
        self.model_error = False
        
        if len(faces_list) == 1:
            try:
                face_crop = crop_and_resize_face(
                    img,
                    faces_list[0],
                    padding_ratio=self.padding_ratio,
                )
                self.prediction = predict_face(face_crop, camera_index=self.camera_index)
            except Exception as e:
                print(f"Prediction error: {e}")
                self.model_error = True
                
        if len(faces_list) == 1 and self.prediction is not None and self.prediction["is_match"]:
            code = self.prediction["display_code"]
            self.recognized_employee_code = code
            self.recognized_confidence = self.prediction["confidence"]
            self.recognized_threshold = self.prediction["match_threshold"]
            
            if self.stable_employee_code == code:
                self.stable_frame_count += 1
            else:
                self.stable_employee_code = code
                self.stable_frame_count = 1
        else:
            self.recognized_employee_code = None
            self.recognized_confidence = None
            self.recognized_threshold = None
            self.stable_employee_code = None
            self.stable_frame_count = 0
                
        for (x, y, w, h) in faces_list:
            if len(faces_list) == 1 and self.prediction is not None:
                if self.prediction["is_match"]:
                    label_text = f"{self.prediction['display_code']} ({self.prediction['confidence']:.2f})"
                    box_color = (0, 255, 0)
                else:
                    label_text = f"Unknown ({self.prediction['confidence']:.2f} > {self.prediction['match_threshold']:.2f})"
                    box_color = (0, 0, 255)
            elif self.model_error:
                label_text = "Model error"
                box_color = (0, 0, 255)
            else:
                label_text = "Face detected"
                box_color = (0, 215, 255)
                
            cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), box_color, 2)
            cv2.putText(
                annotated_frame,
                label_text,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                box_color,
                2,
            )

        cv2.putText(
            annotated_frame,
            f"Faces detected: {len(faces_list)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")
def release_inactive_cameras(*args, **kwargs):
    pass

# file: face_detection/face_detector.py

import cv2
import os

class FaceDetector:
    """
    Haar Cascade ile yüz tespiti yapan sınıf.
    """
    def __init__(self, cascade_path=r"C:\Users\90551\Desktop\new_happy_class\project\face_detection\haarcascade_frontalface_default.xml"):
        """
        cascade_path: Haar Cascade XML dosyasının konumu.
        """
        if not os.path.exists(cascade_path):
            raise FileNotFoundError(f"Haar Cascade dosyası bulunamadı: {cascade_path}")
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

    def detect_faces(self, frame_bgr):
        """
        frame_bgr: OpenCV BGR formatında kare.
        Dönüş: (x, y, w, h) tuple listesi
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        return faces

# file: ui/background_analyzer.py

from PyQt5.QtCore import QThread, pyqtSignal
import pandas as pd
import cv2

from face_detection.face_detector import FaceDetector
from emotion_analysis.emotion_model import predict_emotion

class BackgroundAnalyzer(QThread):
    """
    Arka planda, video dosyası için yüz ve duygu analizi yapan bir QThread.
    'analysis_done' sinyaliyle sonuç DataFrame'i döndürür.
    skip_rate: 1 -> her kareyi oku,
               2 -> 2 kare okuyup sonuncusunda analiz et, vb...
    """
    analysis_done = pyqtSignal(pd.DataFrame)

    def __init__(self, video_path, skip_rate=1, parent=None):
        super().__init__(parent)
        self.video_path = video_path
        self.skip_rate = skip_rate
        self.detector = FaceDetector()
        self.result_data = []
        self.stop_thread = False

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"Video açılamadı: {self.video_path}")
            self.analysis_done.emit(pd.DataFrame())  # Boş DF
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30  # Varsayılan

        frame_count = 0
        emotion_cols = ["Angry", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
        accumulator = {emo: 0 for emo in emotion_cols}

        while True:
            if self.stop_thread:
                break

            # skip_rate kareyi atlayarak ilerliyoruz
            for _ in range(self.skip_rate):
                ret, frame = cap.read()
                if not ret:
                    break

            if not ret:
                break

            # Yüz tespiti
            faces = self.detector.detect_faces(frame)
            for (x, y, w, h) in faces:
                face_region = frame[y:y+h, x:x+w]
                emotion = predict_emotion(face_region)
                if emotion in accumulator:
                    accumulator[emotion] += 1

            frame_count += self.skip_rate

            # Her 1 saniyede bir toplanan duygular DF'e ekleniyor
            if frame_count % int(fps) == 0:
                second_num = frame_count // int(fps)
                data_record = {"second": second_num}
                data_record.update(accumulator)
                self.result_data.append(data_record)
                # yeniden sıfırla
                accumulator = {emo: 0 for emo in emotion_cols}

        cap.release()

        if len(self.result_data) > 0:
            df = pd.DataFrame(self.result_data)
        else:
            df = pd.DataFrame()

        self.analysis_done.emit(df)

    def stop(self):
        """
        Thread'i durdurmak için kullanılabilir.
        """
        self.stop_thread = True

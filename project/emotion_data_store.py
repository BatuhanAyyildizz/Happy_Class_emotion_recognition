# file: emotion_data_store.py

import pandas as pd

class EmotionDataStore:
    """
    Duygu analizi sonuçlarını saklamak için bir yardımcı sınıf.
    Her bir kaydı (face_id, emotion, intensity, timestamp) ile ekler
    ve CSV'ye kaydedebilir.
    """
    def __init__(self):
        self.data = []  # basit liste

    def add_emotion_data(self, face_id, emotion, intensity, timestamp):
        """
        :param face_id: Örneğin 'video_face' veya 'cam_face'
        :param emotion: Tahmin edilen duygu stringi
        :param intensity: Herhangi bir confidence / skor
        :param timestamp: Zaman damgası (örneğin 'video_time')
        """
        self.data.append({
            "Face ID": face_id,
            "Emotion": emotion,
            "Intensity": intensity,
            "Timestamp": timestamp
        })

    def save_to_csv(self, file_name="emotion_data.csv"):
        """
        Kaydedilen tüm verileri tek seferde CSV'ye yazar.
        """
        if not self.data:
            print("Kaydedilecek veri yok!")
            return
        df = pd.DataFrame(self.data)
        df.to_csv(file_name, index=False)
        print(f"Duygu verileri {file_name} dosyasına kaydedildi.")

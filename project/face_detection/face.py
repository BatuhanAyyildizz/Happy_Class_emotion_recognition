# file: face_detection/face.py

class Face:
    """
    Bir yüzün ID'si ve konumu (x, y, w, h) bilgisini tutar.
    Yüz bölgesini kırpmak (extract_features) ve duyguyu atamak için metotlar içerir.
    Şu an kullanılmıyor, ileride genişletilebilir.
    """
    def __init__(self, face_id, position):
        self.face_id = face_id
        self.position = position  # (x, y, w, h)
        self.emotion = None

    def extract_features(self, frame):
        x, y, w, h = self.position
        return frame[y:y + h, x:x + w]

    def set_emotion(self, emotion):
        self.emotion = emotion
# file: face_detection/face.py

class Face:
    """
    Bir yüzün ID'si ve konumu (x, y, w, h) bilgisini tutar.
    Yüz bölgesini kırpmak (extract_features) ve duyguyu atamak için metotlar içerir.
    Şu an kullanılmıyor, ileride genişletilebilir.
    """
    def __init__(self, face_id, position):
        self.face_id = face_id
        self.position = position  # (x, y, w, h)
        self.emotion = None

    def extract_features(self, frame):
        x, y, w, h = self.position
        return frame[y:y + h, x:x + w]

    def set_emotion(self, emotion):
        self.emotion = emotion

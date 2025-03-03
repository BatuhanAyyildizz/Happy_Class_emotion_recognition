
# file: emotion_analysis/emotion_model.py

import os
import cv2
import numpy as np
from tensorflow.keras.models import model_from_json

EMOTIONS = ["Angry", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# Model dosyalarının konumunu ayarlayın (örnek)
json_path = r"C:\Users\90551\Desktop\new_happy_class\project\emotion_analysis\100emotion_model.json"
weights_path = r"C:\Users\90551\Desktop\new_happy_class\project\emotion_analysis\emotion_model_v100.weights.h5"

if not os.path.exists(json_path):
    raise FileNotFoundError(f"Model JSON bulunamadı: {json_path}")

if not os.path.exists(weights_path):
    raise FileNotFoundError(f"Model Weights (.h5) bulunamadı: {weights_path}")

# Model yükleme
with open(json_path, 'r') as json_file:
    loaded_model_json = json_file.read()

emotion_model = model_from_json(loaded_model_json)
emotion_model.load_weights(weights_path)

def preprocess_face(face_bgr):
    """
    BGR'den griye çevir, 48x48'e resize, [0,1] normalleştir,
    (1,48,48,1) boyutunda numpy array döndür.
    """
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (48, 48))
    resized = resized.astype("float32") / 255.0
    resized = np.expand_dims(resized, axis=-1)  # (48,48,1)
    resized = np.expand_dims(resized, axis=0)   # (1,48,48,1)
    return resized

def predict_emotion(face_bgr):
    """
    Yüz bölgesini modelin beklediği formata sokar,
    duyguyu tahmin eder, EMOTIONS listesinden etiketi döndürür.
    """
    try:
        input_data = preprocess_face(face_bgr)
        predictions = emotion_model.predict(input_data)
        idx = np.argmax(predictions)
        return EMOTIONS[idx]
    except Exception as e:
        print(f"predict_emotion hata: {e}")
        return "Unknown"

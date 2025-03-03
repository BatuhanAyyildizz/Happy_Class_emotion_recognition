# file: tests/test_app.py

import os
import pytest
import cv2
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import QApplication

# Projenizdeki modüller
from ui.graph_widget import GraphWidget
from emotion_data_store import EmotionDataStore
from face_detection.face_detector import FaceDetector
from emotion_analysis.emotion_model import preprocess_face, predict_emotion, EMOTIONS
from ui.background_analyzer import BackgroundAnalyzer

# PyQt uygulamaları için pytest plugin veya QApplication nesnesine ihtiyaç duyulabilir
# Normalde testler için bir QApplication örneği gerekli olabilir:
app = QApplication([])  # test sırasında GUI bileşenlerini oluşturmak için

# =========================================
#  1) GRAPH WIDGET TESTLERİ (4 ADET)
# =========================================

def test_graph_widget_initial_counts():
    """
    GraphWidget oluşturulduğunda tüm duyguların 0 olması beklenir.
    """
    gw = GraphWidget()
    for emo, val in gw.emotion_counts.items():
        assert val == 0, f"{emo} başlangıç değeri 0 olmalı, fakat {val} bulundu."

def test_graph_widget_update_counts_valid():
    """
    Var olan bir duygu etiketini gönderdiğimizde sayım 1 artmalı.
    """
    gw = GraphWidget()
    gw.update_counts("Happy")
    assert gw.emotion_counts["Happy"] == 1, "Happy sayacı 1 olmalıydı."

def test_graph_widget_update_counts_invalid():
    """
    Geçersiz bir duygu etiketi gönderdiğimizde grafik güncellenmemeli.
    """
    gw = GraphWidget()
    gw.update_counts("InvalidEmotion")
    # Tüm duyguların hâlâ 0 olduğu kontrol edelim
    assert all(v == 0 for v in gw.emotion_counts.values()), \
        "Geçersiz etiketle hiçbir duygu artmamalıydı."

def test_graph_widget_reset_counts():
    """
    reset_counts() çağrıldığında tüm değerler yeniden 0 olmalı.
    """
    gw = GraphWidget()
    gw.update_counts("Angry")
    gw.update_counts("Happy")
    gw.reset_counts()
    for emo, val in gw.emotion_counts.items():
        assert val == 0, f"{emo} reset sonrası 0 olmalı, {val} bulundu."

# =========================================
#  2) EMOTIONDATASTORE TESTLERİ (3 ADET)
# =========================================

def test_emotion_data_store_add():
    """
    add_emotion_data metodu ile eklenen veriler 'data' listesinde saklanmalı.
    """
    store = EmotionDataStore()
    store.add_emotion_data("video_face", "Happy", 0.9, "video_time")
    assert len(store.data) == 1, "Data listesine 1 kayıt eklenmeli."
    record = store.data[0]
    assert record["Face ID"] == "video_face"
    assert record["Emotion"] == "Happy"
    assert record["Intensity"] == 0.9
    assert record["Timestamp"] == "video_time"

def test_emotion_data_store_save_to_csv_no_data(tmp_path):
    """
    Hiç veri yokken save_to_csv çağrıldığında 'Kaydedilecek veri yok!' yazmalı 
    ve dosya oluşmamalı.
    """
    store = EmotionDataStore()
    csv_path = tmp_path / "test_no_data.csv"
    store.save_to_csv(str(csv_path))
    assert not csv_path.exists(), "Veri yokken CSV dosyası oluşmamalı."

def test_emotion_data_store_save_to_csv_with_data(tmp_path):
    """
    Veri varsa CSV dosyası başarıyla yazılmalı.
    """
    store = EmotionDataStore()
    store.add_emotion_data("video_face", "Sad", 1.0, "video_time")
    csv_path = tmp_path / "test_data.csv"
    store.save_to_csv(str(csv_path))
    assert csv_path.exists(), "CSV dosyası oluşturulmalı."
    df = pd.read_csv(csv_path)
    assert len(df) == 1, "CSV'de 1 satır olmalı."
    assert df.iloc[0]["Emotion"] == "Sad", "Eklenen duygu 'Sad' bulunamadı."

# =========================================
#  3) FACEDETECTOR TESTLERİ (2 ADET)
# =========================================

def test_face_detector_init():
    """
    Cascade path yanlışsa FileNotFoundError atmalı, 
    doğruysa sınıf örneği oluşturabilmeli.
    """
    # Yanlış path deneme
    with pytest.raises(FileNotFoundError):
        FaceDetector(cascade_path="face_detection/does_not_exist.xml")

    # Doğru path - not: Bu testi çalıştırmak için gerçek cascade path'inizin var olması gerek.
    # Örneğin cascade_path'i projenizdeki tam path'e göre değiştirin.
    # Cascade dosyası yoksa bu test fail eder (veya skip).
    correct_path = "face_detection/haarcascade_frontalface_default.xml"
    if os.path.exists(correct_path):
        detector = FaceDetector(cascade_path=correct_path)
        assert isinstance(detector, FaceDetector), "FaceDetector örneği oluşturulamadı."

def test_face_detector_detect_faces():
    """
    Basit bir gri resim oluşturup yüz tespitini test edebiliriz.
    Burada gerçekte yüz olmasa bile hata vermeden boş list dönmeli.
    """
    correct_path = "face_detection/haarcascade_frontalface_default.xml"
    if not os.path.exists(correct_path):
        pytest.skip("Cascade dosyası yok, bu test atlanıyor.")

    detector = FaceDetector(cascade_path=correct_path)
    fake_image = np.zeros((100, 100, 3), dtype=np.uint8)  # siyah bir kare
    faces = detector.detect_faces(fake_image)
    assert isinstance(faces, tuple) or isinstance(faces, list), "Dönen değer tuple veya list olmalı."
    assert len(faces) == 0, "Hiç yüz olmayan resimde yüz tespiti olmamalı."

# =========================================
#  4) EMOTION_MODEL TESTLERİ (2 ADET)
# =========================================

def test_preprocess_face():
    """
    preprocess_face metodu 48x48 boyutunda, tek kanallı, 4D array döndürmeli.
    """
    fake_face = np.ones((60, 60, 3), dtype=np.uint8) * 255  # beyaz 60x60 RGB
    processed = preprocess_face(fake_face)
    # Beklenen şekil: (1, 48, 48, 1)
    assert processed.shape == (1, 48, 48, 1), f"Beklenen (1,48,48,1), bulundu {processed.shape}."

def test_predict_emotion_unknown():
    """
    Çok küçük veya geçersiz bir resim girdiğimizde 'Unknown' dönebilir.
    """
    # 10x10'luk çok küçük bir resim
    tiny_face = np.zeros((10, 10, 3), dtype=np.uint8)
    pred = predict_emotion(tiny_face)
    # Model hata atabilir, try/except ile 'Unknown' dönebiliyor.
    # Burada 'Unknown' bekliyoruz. 
    assert pred == "Unknown", f"Küçük resimde 'Unknown' bekleniyordu, '{pred}' geldi."

# =========================================
#  5) BACKGROUNDANALYZER TESTLERİ (1 ADET)
# =========================================

def test_background_analyzer_no_video(tmp_path):
    """
    Var olmayan bir video path'i verildiğinde 
    analysis_done sinyali boş bir DataFrame göndermeli.
    """
    analyzer = BackgroundAnalyzer("non_existent_video.mp4")
    results = []

    # sinyal yakalamak için basit fonksiyon
    def on_analysis_done(df):
        results.append(df)

    analyzer.analysis_done.connect(on_analysis_done)
    analyzer.run()  # thread yerine direkt run() çağırabiliriz testte

    assert len(results) == 1, "analysis_done sinyali 1 kez emit edilmeli."
    df_result = results[0]
    assert df_result.empty, "Video açılamadığı için DF boş olmalı."

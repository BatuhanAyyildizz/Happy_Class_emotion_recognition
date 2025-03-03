# file: ui/main_window.py

import os
import cv2
import pandas as pd

from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap, QMovie
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QStatusBar
)

from ui.background_analyzer import BackgroundAnalyzer
from face_detection.face_detector import FaceDetector
from emotion_analysis.emotion_model import predict_emotion
from emotion_data_store import EmotionDataStore
from ui.graph_widget import GraphWidget

class EmotionAnalyzerApp(QMainWindow):
    """
    Ana pencere sınıfı:
      - Splash ekranı (büyük GIF)
      - Video seçme/oynatma/duraklatma/durdurma (analiz dahil)
      - Kamera başlat/durdur (canlı analiz + kaydetme)
      - Arka planda analiz
      - Canlı pyqtgraph ile duygu sayımı
      - Tüm verileri CSV'ye kaydetme
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Emotion Analyzer - Gelişmiş Versiyon (PyQtGraph ile)")
        self.setGeometry(100, 100, 1200, 700)

        # Splash ekranı için
        self.splash_label = None
        self.movie = None

        # Kamera/Video
        self.camera_cap = None
        self.camera_timer = None
        self.camera_running = False

        self.video_cap = None
        self.video_path = None
        self.video_timer = None
        self.video_playing = False

        # Arka plan analiz
        self.analyzer_thread = None
        self.analysis_df = None

        # Yüz dedektörü
        self.detector = FaceDetector()

        # Veri deposu
        self.emotion_data_store = EmotionDataStore()

        # PyQtGraph widget (canlı grafik)
        self.graph_widget = None

        # Başlat
        self.init_splash_screen()

    # =====================
    #    Splash Screen
    # =====================
    def init_splash_screen(self):
        self.splash_label = QLabel(self)
        self.splash_label.setGeometry(0, 0, 1200, 700)
        self.splash_label.setScaledContents(True)

        gif_path = r"C:\Users\90551\Desktop\new_happy_class\project\assets\splash.gif"  # kendi yolunuza göre düzenleyin
        if os.path.exists(gif_path):
            self.movie = QMovie(gif_path)
            self.movie.setScaledSize(self.splash_label.size())
            self.splash_label.setMovie(self.movie)
            self.movie.start()
        else:
            self.splash_label.setText("SPLASH GIF BULUNAMADI")
            self.splash_label.setAlignment(Qt.AlignCenter)

        # 3 saniye sonra init_ui
        QTimer.singleShot(3000, self.init_ui)

    # =====================
    #    Ana UI
    # =====================
    def init_ui(self):
        # Splash'i kapat
        if self.splash_label:
            self.splash_label.hide()
            if self.movie:
                self.movie.stop()

        # Ana widget/layout
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        main_layout = QVBoxLayout(self.central_widget)

        # Üstteki butonlar
        button_layout = QHBoxLayout()

        # Video Butonları
        self.btn_open_video = QPushButton("Video Seç")
        self.btn_open_video.clicked.connect(self.open_video_file)
        button_layout.addWidget(self.btn_open_video)

        self.btn_play_video = QPushButton("Play Video")
        self.btn_play_video.clicked.connect(self.play_video)
        self.btn_play_video.setEnabled(False)
        button_layout.addWidget(self.btn_play_video)

        self.btn_pause_video = QPushButton("Pause Video")
        self.btn_pause_video.clicked.connect(self.pause_video)
        self.btn_pause_video.setEnabled(False)
        button_layout.addWidget(self.btn_pause_video)

        self.btn_stop_video = QPushButton("Stop Video")
        self.btn_stop_video.clicked.connect(self.stop_video)
        self.btn_stop_video.setEnabled(False)
        button_layout.addWidget(self.btn_stop_video)

        # Kamera Butonları
        self.btn_start_cam = QPushButton("Kamerayı Başlat")
        self.btn_start_cam.clicked.connect(self.start_camera)
        button_layout.addWidget(self.btn_start_cam)

        self.btn_stop_cam = QPushButton("Kamerayı Durdur")
        self.btn_stop_cam.clicked.connect(self.stop_camera)
        self.btn_stop_cam.setEnabled(False)
        button_layout.addWidget(self.btn_stop_cam)

        # Arka Planda Analiz
        self.btn_bg_analyze = QPushButton("Arka Planda Analiz")
        self.btn_bg_analyze.clicked.connect(self.start_background_analysis)
        button_layout.addWidget(self.btn_bg_analyze)

        # Analiz Sonucu Göster
        self.btn_show_analysis = QPushButton("Analiz Sonuçlarını Göster")
        self.btn_show_analysis.clicked.connect(self.show_analysis_results)
        self.btn_show_analysis.setEnabled(False)
        button_layout.addWidget(self.btn_show_analysis)

        # Butonları ana layout'a ekle
        main_layout.addLayout(button_layout)

        # Orta kısımda kamera/video görüntüsü
        self.display_label = QLabel("Görüntü burada gösterilecek.")
        self.display_label.setAlignment(Qt.AlignCenter)

        # Alt kısımda canlı grafik
        self.graph_widget = GraphWidget()

        # Hepsini main_layout'a ekle
        main_layout.addWidget(self.display_label)
        main_layout.addWidget(self.graph_widget)

        # Status bar
        self.setStatusBar(QStatusBar(self))

    # ===========================
    #   Video Seç & Oynatma
    # ===========================
    def open_video_file(self):
        file_dialog = QFileDialog()
        video_path, _ = file_dialog.getOpenFileName(
            self, "Bir video seçin", "", "Video Files (*.mp4 *.avi *.mov)"
        )
        if video_path:
            self.video_path = video_path
            self.statusBar().showMessage(f"Seçilen video: {video_path}")
            self.btn_play_video.setEnabled(True)
            self.btn_pause_video.setEnabled(False)
            self.btn_stop_video.setEnabled(False)

            # Bir önceki grafiği sıfırlamak isterseniz:
            self.graph_widget.reset_counts()

    def play_video(self):
        if not self.video_path:
            return

        # varsa eski capture kapat
        if self.video_cap and self.video_cap.isOpened():
            self.video_cap.release()

        self.video_cap = cv2.VideoCapture(self.video_path)
        if not self.video_cap.isOpened():
            self.statusBar().showMessage("Video açılamadı!")
            return

        self.video_playing = True
        self.btn_play_video.setEnabled(False)
        self.btn_pause_video.setEnabled(True)
        self.btn_stop_video.setEnabled(True)

        # Timer
        self.video_timer = QTimer()
        self.video_timer.timeout.connect(self.update_video_frame)
        self.video_timer.start(30)

    def pause_video(self):
        if self.video_timer:
            self.video_timer.stop()
        self.video_playing = False
        self.btn_play_video.setEnabled(True)
        self.btn_pause_video.setEnabled(False)
        self.btn_stop_video.setEnabled(True)
        self.statusBar().showMessage("Video duraklatıldı.")

    def stop_video(self):
        if self.video_timer:
            self.video_timer.stop()
            self.video_timer = None

        if self.video_cap and self.video_cap.isOpened():
            self.video_cap.release()

        self.video_playing = False
        self.display_label.setText("Video durduruldu.")
        self.btn_play_video.setEnabled(True)
        self.btn_pause_video.setEnabled(False)
        self.btn_stop_video.setEnabled(False)
        self.statusBar().showMessage("Video durduruldu.")

    def update_video_frame(self):
        if not (self.video_cap and self.video_cap.isOpened()):
            return

        ret, frame = self.video_cap.read()
        if not ret:
            # video bitti
            self.stop_video()
            return

        faces = self.detector.detect_faces(frame)
        for (x, y, w, h) in faces:
            face_region = frame[y:y+h, x:x+w]
            emotion_label = predict_emotion(face_region)

            # karede çizim
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, emotion_label, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # CSV için kaydet
            self.emotion_data_store.add_emotion_data(
                face_id="video_face",
                emotion=emotion_label,
                intensity=1.0,
                timestamp="video_time"
            )

            # Gerçek zamanlı grafik güncelle
            self.graph_widget.update_counts(emotion_label)

        # Ekrana göster (PyQt)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h,
                          bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        self.display_label.setPixmap(pixmap)
        self.display_label.setScaledContents(True)

    # ===========================
    #   Kamera İşlemleri
    # ===========================
    def start_camera(self):
        # varsa eski capture kapat
        if self.camera_cap and self.camera_cap.isOpened():
            self.camera_cap.release()

        self.camera_cap = cv2.VideoCapture(0)
        if not self.camera_cap.isOpened():
            self.statusBar().showMessage("Kamera açılamadı!")
            return

        self.camera_running = True
        self.btn_stop_cam.setEnabled(True)
        self.btn_start_cam.setEnabled(False)

        # Kamerayı açtığımızda grafiği sıfırlayabiliriz (isteğe bağlı)
        self.graph_widget.reset_counts()

        self.camera_timer = QTimer()
        self.camera_timer.timeout.connect(self.update_camera_frame)
        self.camera_timer.start(30)
        self.statusBar().showMessage("Kamera başlatıldı.")

    def stop_camera(self):
        self.camera_running = False
        self.btn_start_cam.setEnabled(True)
        self.btn_stop_cam.setEnabled(False)

        if self.camera_timer:
            self.camera_timer.stop()
            self.camera_timer = None

        if self.camera_cap and self.camera_cap.isOpened():
            self.camera_cap.release()

        self.display_label.setText("Kamera durduruldu.")
        self.statusBar().showMessage("Kamera durduruldu.")

        # Kamera sırasındaki verileri CSV'ye yazmak isterseniz
        self.emotion_data_store.save_to_csv("camera_analysis.csv")

    def update_camera_frame(self):
        if not (self.camera_cap and self.camera_cap.isOpened()):
            return

        ret, frame = self.camera_cap.read()
        if not ret:
            return

        # Yüz tespiti
        faces = self.detector.detect_faces(frame)
        for (x, y, w, h) in faces:
            face_region = frame[y:y+h, x:x+w]
            emotion_label = predict_emotion(face_region)

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, emotion_label, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Depoya ekle
            self.emotion_data_store.add_emotion_data(
                face_id="cam_face",
                emotion=emotion_label,
                intensity=1.0,
                timestamp="camera_time"
            )

            # Grafik güncelle
            self.graph_widget.update_counts(emotion_label)

        # Görüntüyü ekranda göster
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h,
                          bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        self.display_label.setPixmap(pixmap)
        self.display_label.setScaledContents(True)

    # ==================================
    #   Arka Planda Analiz (QThread)
    # ==================================
    def start_background_analysis(self):
        file_dialog = QFileDialog()
        video_path, _ = file_dialog.getOpenFileName(
            self, "Arka planda analiz için video seçin", "",
            "Video Files (*.mp4 *.avi *.mov)"
        )
        if not video_path:
            return

        self.analyzer_thread = BackgroundAnalyzer(video_path, skip_rate=2)
        self.analyzer_thread.analysis_done.connect(self.handle_analysis_done)
        self.analyzer_thread.start()
        self.statusBar().showMessage("Arka planda analiz başlatıldı...")

    def handle_analysis_done(self, df):
        self.analysis_df = df
        self.statusBar().showMessage("Arka plan analizi tamamlandı.")
        self.btn_show_analysis.setEnabled(True)

        # CSV kaydet
        csv_path = "analysis_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"Analiz sonuçları '{csv_path}' olarak kaydedildi.")

    def show_analysis_results(self):
        """
        Arka plan analizi sonuçlarını göstermek.
        Artık matplotlib yerine ya pyqtgraph ile de çizebilirsiniz;
        veya gelen df ile "toplam sayıları" hesaplayıp graph_widget'a yansıtabilirsiniz.
        Burada basit bir örnek, eğer isterseniz graph_widget reset edip
        df'in total'larını yansıtabilirsiniz.
        """
        if self.analysis_df is None or self.analysis_df.empty:
            self.statusBar().showMessage("Gösterilecek analiz verisi yok.")
            return

        # Örnek: df içindeki toplamları alalım
        sum_emotions = self.analysis_df[["Angry","Fear","Happy","Sad","Surprise","Neutral"]].sum()
        # Grafiği sıfırla
        self.graph_widget.reset_counts()

        # Her duyguyu graph_widget'a set edelim
        for emo, val in sum_emotions.items():
            # barların yüksekliğini val kadar yapsın
            self.graph_widget.emotion_counts[emo] = int(val)

        self.graph_widget._update_bar_graph()
        self.statusBar().showMessage("Analiz sonuçları grafiğe aktarıldı.")

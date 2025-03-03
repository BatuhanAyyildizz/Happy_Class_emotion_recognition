# file: ui/graph_widget.py

import pyqtgraph as pg
from pyqtgraph import BarGraphItem
from PyQt5.QtWidgets import QWidget, QVBoxLayout

class GraphWidget(QWidget):
    """
    PyQtGraph kullanarak dinamik (near real-time) bar chart gösteren bir widget.
    Duygu sayımlarını (Angry, Fear, Happy, ...) güncelleyerek çubukları yeniler.
    """
    def __init__(self, parent=None):
        super().__init__(parent)

        self.emotions = ["Angry", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
        # Başlangıçta tüm duyguların sayısı sıfır olsun
        self.emotion_counts = {emo: 0 for emo in self.emotions}

        # PyQtGraph PlotWidget
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('w')  # beyaz arkaplan
        self.plot_widget.showGrid(x=True, y=True)
        self.plot_widget.setTitle("Real-Time Emotion Counts")
        self.plot_widget.setLabel('left', 'Count')
        self.plot_widget.setLabel('bottom', 'Emotion')

        layout = QVBoxLayout()
        layout.addWidget(self.plot_widget)
        self.setLayout(layout)

        # Bar chart item
        self.bar_item = None
        self._init_bars()

    def _init_bars(self):
        """
        Bar grafinin ilk ayarlarını yapar.
        """
        x_positions = range(len(self.emotions))
        heights = [0] * len(self.emotions)
        self.bar_item = BarGraphItem(
            x=x_positions, height=heights, width=0.6,
            brush='blue'
        )
        self.plot_widget.addItem(self.bar_item)
        # X eksenine emotion etiketleri
        ax = self.plot_widget.getAxis('bottom')
        ticks = [(i, emo) for i, emo in enumerate(self.emotions)]
        ax.setTicks([ticks])

    def update_counts(self, new_emotion):
        """
        Belirli bir duyguyu tespit ettiğimizde, sayımı artırıp grafiği günceller.
        """
        if new_emotion not in self.emotion_counts:
            return
        self.emotion_counts[new_emotion] += 1
        self._update_bar_graph()

    def reset_counts(self):
        """
        Yeni video/kamera için grafiği sıfırlamak isterseniz.
        """
        for emo in self.emotions:
            self.emotion_counts[emo] = 0
        self._update_bar_graph()

    def _update_bar_graph(self):
        """
        BarGraphItem'ı günceller.
        """
        x_positions = range(len(self.emotions))
        heights = [self.emotion_counts[emo] for emo in self.emotions]
        self.bar_item.setOpts(x=x_positions, height=heights)

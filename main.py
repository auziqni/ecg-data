import sys
import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QHBoxLayout, QLabel, QComboBox, QCheckBox
from PyQt5.QtCore import QTimer
import wfdb
import os

class ECGMonitor(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Parameter plot
        self.window_size = 2000  # Jumlah sampel yang ditampilkan pada satu waktu
        self.sample_rate = 360   # Sample rate data ECG (biasanya 360 Hz untuk MIT-BIH)
        self.current_index = 0   # Indeks awal untuk data
        self.playing = False     # Status pemutaran
        
        # Data PhysioNet
        self.record_name = '100'  # Record default dari MIT-BIH
        self.record_path = None   # Path ke file record
        self.signal = None        # Data sinyal
        self.time = None          # Waktu (dalam detik)
        self.show_channel = [True, True]  # Menampilkan kedua channel
        
        # Pengaturan window
        self.setWindowTitle("PhysioNet ECG Monitor - Multi-Channel")
        self.setGeometry(100, 100, 1200, 700)
        
        # Membuat widget utama dan layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        
        # Panel kontrol
        self.control_layout = QHBoxLayout()
        self.main_layout.addLayout(self.control_layout)
        
        # Tombol Play/Pause
        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self.toggle_play)
        self.control_layout.addWidget(self.play_button)
        
        # Pemilihan record
        self.record_label = QLabel("Record:")
        self.control_layout.addWidget(self.record_label)
        
        self.record_combo = QComboBox()
        # Beberapa record dari MIT-BIH
        self.record_combo.addItems(['100', '101', '102', '103', '104', '105'])
        self.record_combo.currentTextChanged.connect(self.load_record)
        self.control_layout.addWidget(self.record_combo)
        
        # Checkbox untuk menampilkan channel
        self.channel0_check = QCheckBox("Channel 0")
        self.channel0_check.setChecked(True)
        self.channel0_check.stateChanged.connect(lambda state: self.toggle_channel(0, state))
        self.control_layout.addWidget(self.channel0_check)
        
        self.channel1_check = QCheckBox("Channel 1")
        self.channel1_check.setChecked(True)
        self.channel1_check.stateChanged.connect(lambda state: self.toggle_channel(1, state))
        self.control_layout.addWidget(self.channel1_check)
        
        # Informasi status
        self.status_label = QLabel("Status: Siap")
        self.control_layout.addWidget(self.status_label)
        
        # Membuat plot widget
        self.plot_widget = pg.PlotWidget()
        self.main_layout.addWidget(self.plot_widget)
        
        # Pengaturan plot
        self.plot_widget.setBackground('w')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setLabel('left', 'Amplitudo (mV)')
        self.plot_widget.setLabel('bottom', 'Waktu (detik)')
        self.plot_widget.setTitle('ECG Signal Monitor - Multi-Channel')
        
        # Membuat dua plot line untuk dua channel
        self.plot_lines = [
            self.plot_widget.plot(pen=pg.mkPen(color=(255, 0, 0), width=2), name="Channel 0"),
            self.plot_widget.plot(pen=pg.mkPen(color=(0, 0, 255), width=2), name="Channel 1")
        ]
        
        # Menambahkan legenda
        self.plot_widget.addLegend()
        
        # Timer untuk update plot
        self.timer = QTimer()
        self.timer.setInterval(25)  # 40 fps, cukup smooth untuk visualisasi
        self.timer.timeout.connect(self.update_plot)
        
        # Memuat data record default
        self.load_record(self.record_name)
        
    def load_record(self, record_name):
        try:
            # Download file dari PhysioNet jika belum ada
            if not os.path.exists(f'{record_name}.dat'):
                self.status_label.setText(f"Status: Mengunduh record {record_name}...")
                # Dalam praktiknya, Anda perlu mengunduh file dari PhysioNet
                # Gunakan wfdb.dl_database atau urllib.request
                
            # Memuat record
            self.status_label.setText(f"Status: Memuat record {record_name}...")
            self.record = wfdb.rdrecord(record_name)
            self.signal = self.record.p_signal
            self.sample_rate = self.record.fs
            
            # Membuat array waktu
            self.time = np.arange(len(self.signal)) / self.sample_rate
            
            # Reset indeks
            self.current_index = 0
            
            # Update checkbox label dengan nama channel
            if hasattr(self.record, 'sig_name') and len(self.record.sig_name) >= 2:
                self.channel0_check.setText(f"Channel 0 ({self.record.sig_name[0]})")
                self.channel1_check.setText(f"Channel 1 ({self.record.sig_name[1]})")
            
            # Update status
            self.status_label.setText(f"Status: Record {record_name} dimuat")
            
            # Update plot satu kali
            self.update_plot_single()
            
        except Exception as e:
            self.status_label.setText(f"Error: {str(e)}")
    
    def toggle_channel(self, channel, state):
        self.show_channel[channel] = state > 0
        # Update visibility plot
        self.plot_lines[channel].setVisible(self.show_channel[channel])
        self.update_plot_single()
    
    def toggle_play(self):
        if self.playing:
            self.timer.stop()
            self.playing = False
            self.play_button.setText("Play")
        else:
            self.timer.start()
            self.playing = True
            self.play_button.setText("Pause")
    
    def update_plot_single(self):
        # Menampilkan window data saat ini tanpa increment
        if self.signal is not None and len(self.signal) > 0:
            end_idx = min(self.current_index + self.window_size, len(self.signal))
            visible_time = self.time[self.current_index:end_idx]
            
            # Siapkan untuk perhitungan auto-scale
            all_visible_data = []
            
            # Update setiap channel
            for i in range(min(2, self.signal.shape[1])):
                if self.show_channel[i]:
                    visible_data = self.signal[self.current_index:end_idx, i]
                    self.plot_lines[i].setData(visible_time, visible_data)
                    all_visible_data.extend(visible_data)
            
            # Auto-scale y-axis jika ada data yang visible
            if all_visible_data:
                min_val = np.min(all_visible_data) - 0.1
                max_val = np.max(all_visible_data) + 0.1
                self.plot_widget.setYRange(min_val, max_val)
            
            # Update x range untuk scrolling
            self.plot_widget.setXRange(visible_time[0], visible_time[-1])
    
    def update_plot(self):
        # Menambah indeks untuk efek scrolling
        if self.signal is not None and len(self.signal) > 0:
            # Increment indeks (kecepatan scrolling)
            increment = max(1, int(self.window_size * 0.1))
            self.current_index += increment
            
            # Jika mencapai akhir sinyal, kembali ke awal
            if self.current_index >= len(self.signal) - self.window_size:
                self.current_index = 0
            
            # Update plot
            self.update_plot_single()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ECGMonitor()
    window.show()
    sys.exit(app.exec_())
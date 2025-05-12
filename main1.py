import sys
import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QWidget, 
                            QPushButton, QHBoxLayout, QLabel, QComboBox, 
                            QCheckBox, QSlider, QSpinBox, QGridLayout, QGroupBox)
from PyQt5.QtCore import QTimer, Qt
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
        self.play_speed = 1.0    # Faktor kecepatan, 1.0 adalah kecepatan normal
        
        # Data PhysioNet
        self.record_name = '100'  # Record default dari MIT-BIH
        self.record_path = None   # Path ke file record
        self.signal = None        # Data sinyal
        self.time = None          # Waktu (dalam detik)
        self.show_channel = [True, True]  # Menampilkan kedua channel
        
        # Pengaturan window
        self.setWindowTitle("PhysioNet ECG Monitor - Real-time Visualization")
        self.setGeometry(100, 100, 1200, 800)
        
        # Membuat widget utama dan layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        
        # === Panel Kontrol ===
        self.create_control_panel()
        
        # === Plot Widget ===
        self.create_plot_widget()
        
        # Timer untuk update plot
        self.timer = QTimer()
        self.timer.setInterval(25)  # Default 25ms (40 fps)
        self.timer.timeout.connect(self.update_plot)
        
        # Status bar
        self.statusBar().showMessage("Siap. Silakan pilih record dan klik Play.")
        
        # Memuat data record default
        self.load_record(self.record_name)
    
    def create_control_panel(self):
        # Panel kontrol dalam group box
        self.control_group = QGroupBox("Kontrol")
        self.main_layout.addWidget(self.control_group)
        self.control_layout = QGridLayout(self.control_group)
        
        # === Record Control ===
        # Tombol Play/Pause
        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self.toggle_play)
        self.control_layout.addWidget(self.play_button, 0, 0)
        
        # Tombol Reset
        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self.reset_playback)
        self.control_layout.addWidget(self.reset_button, 0, 1)
        
        # Pemilihan record
        self.record_label = QLabel("Record:")
        self.control_layout.addWidget(self.record_label, 0, 2)
        
        self.record_combo = QComboBox()
        # Beberapa record dari MIT-BIH
        self.record_combo.addItems(['100', '101', '102', '103', '104', '105', '106', '107', '108', '109'])
        self.record_combo.currentTextChanged.connect(self.load_record)
        self.control_layout.addWidget(self.record_combo, 0, 3)
        
        # === Channel Control ===
        # Checkbox untuk menampilkan channel
        self.channel_label = QLabel("Channels:")
        self.control_layout.addWidget(self.channel_label, 0, 4)
        
        self.channel0_check = QCheckBox("Channel 0")
        self.channel0_check.setChecked(True)
        self.channel0_check.stateChanged.connect(lambda state: self.toggle_channel(0, state))
        self.control_layout.addWidget(self.channel0_check, 0, 5)
        
        self.channel1_check = QCheckBox("Channel 1")
        self.channel1_check.setChecked(True)
        self.channel1_check.stateChanged.connect(lambda state: self.toggle_channel(1, state))
        self.control_layout.addWidget(self.channel1_check, 0, 6)
        
        # === Speed Control ===
        # Speed label
        self.speed_label = QLabel("Kecepatan:")
        self.control_layout.addWidget(self.speed_label, 1, 0)
        
        # Slider kecepatan
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setMinimum(1)   # 0.1x
        self.speed_slider.setMaximum(50)  # 5.0x
        self.speed_slider.setValue(10)    # 1.0x (default)
        self.speed_slider.setTickPosition(QSlider.TicksBelow)
        self.speed_slider.setTickInterval(5)
        self.speed_slider.valueChanged.connect(self.change_speed)
        self.control_layout.addWidget(self.speed_slider, 1, 1, 1, 3)
        
        # Display nilai kecepatan
        self.speed_value = QLabel("1.0x")
        self.control_layout.addWidget(self.speed_value, 1, 4)
        
        # === Interval Control ===
        # Interval label
        self.interval_label = QLabel("Interval Update (ms):")
        self.control_layout.addWidget(self.interval_label, 1, 5)
        
        # Spinbox untuk interval
        self.interval_spinbox = QSpinBox()
        self.interval_spinbox.setMinimum(10)
        self.interval_spinbox.setMaximum(500)
        self.interval_spinbox.setValue(25)  # Default 25ms
        self.interval_spinbox.valueChanged.connect(self.change_interval)
        self.control_layout.addWidget(self.interval_spinbox, 1, 6)
        
        # === Window Size Control ===
        # Window size label
        self.window_label = QLabel("Window Size (sampel):")
        self.control_layout.addWidget(self.window_label, 2, 0)
        
        # Spinbox untuk window size
        self.window_spinbox = QSpinBox()
        self.window_spinbox.setMinimum(500)
        self.window_spinbox.setMaximum(10000)
        self.window_spinbox.setSingleStep(500)
        self.window_spinbox.setValue(2000)  # Default 2000 sampel
        self.window_spinbox.valueChanged.connect(self.change_window_size)
        self.control_layout.addWidget(self.window_spinbox, 2, 1)
        
        # Informasi durasi window
        self.window_duration = QLabel("(5.56 detik)")
        self.control_layout.addWidget(self.window_duration, 2, 2)
        
        # Informasi sampel rate
        self.sample_rate_label = QLabel("Sample Rate: 360 Hz")
        self.control_layout.addWidget(self.sample_rate_label, 2, 3, 1, 2)
        
        # Informasi durasi total
        self.duration_label = QLabel("Durasi Total: 0:00")
        self.control_layout.addWidget(self.duration_label, 2, 5, 1, 2)
    
    def create_plot_widget(self):
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
    
    def load_record(self, record_name):
        try:
            self.statusBar().showMessage(f"Memuat record {record_name}...")
            
            # Download file dari PhysioNet jika belum ada
            if not os.path.exists(f'{record_name}.dat'):
                self.statusBar().showMessage(f"File {record_name}.dat tidak ditemukan. Silakan download dari PhysioNet terlebih dahulu.")
                return
                
            # Memuat record
            self.record = wfdb.rdrecord(record_name)
            self.signal = self.record.p_signal
            self.sample_rate = self.record.fs
            
            # Membuat array waktu
            self.time = np.arange(len(self.signal)) / self.sample_rate
            
            # Reset indeks dan update informasi
            self.current_index = 0
            self.update_record_info()
            
            # Update plot satu kali
            self.update_plot_single()
            
            # Update status
            self.statusBar().showMessage(f"Record {record_name} berhasil dimuat.")
            
        except Exception as e:
            self.statusBar().showMessage(f"Error: {str(e)}")
    
    def update_record_info(self):
        if self.signal is None or len(self.signal) == 0:
            return
            
        # Update checkbox label dengan nama channel
        if hasattr(self.record, 'sig_name') and len(self.record.sig_name) >= 2:
            self.channel0_check.setText(f"Channel 0 ({self.record.sig_name[0]})")
            self.channel1_check.setText(f"Channel 1 ({self.record.sig_name[1]})")
        
        # Update sample rate info
        self.sample_rate_label.setText(f"Sample Rate: {self.sample_rate} Hz")
        
        # Update window duration info
        duration_sec = self.window_size / self.sample_rate
        self.window_duration.setText(f"({duration_sec:.2f} detik)")
        
        # Update total duration info
        total_duration = len(self.signal) / self.sample_rate
        minutes = int(total_duration // 60)
        seconds = int(total_duration % 60)
        self.duration_label.setText(f"Durasi Total: {minutes}:{seconds:02d}")
    
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
    
    def reset_playback(self):
        self.current_index = 0
        self.update_plot_single()
        self.statusBar().showMessage("Playback direset ke awal.")
    
    def change_speed(self, value):
        self.play_speed = value / 10.0  # Konversi (1-50) menjadi (0.1-5.0)
        self.speed_value.setText(f"{self.play_speed:.1f}x")
        
        # Jika sedang diputar, restart timer dengan interval baru
        if self.playing:
            self.timer.stop()
            self.timer.start()
    
    def change_interval(self, value):
        self.timer.setInterval(value)
        
        # Jika sedang diputar, restart timer dengan interval baru
        if self.playing:
            self.timer.stop()
            self.timer.start()
    
    def change_window_size(self, value):
        self.window_size = value
        self.update_record_info()
        self.update_plot_single()
    
    def update_plot_single(self):
        # Menampilkan window data saat ini tanpa increment
        if self.signal is None or len(self.signal) == 0:
            return
            
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
            padding = (max_val - min_val) * 0.1  # 10% padding
            self.plot_widget.setYRange(min_val - padding, max_val + padding)
        
        # Update x range untuk scrolling
        if len(visible_time) > 0:
            self.plot_widget.setXRange(visible_time[0], visible_time[-1])
            
        # Update status dengan posisi waktu
        if len(visible_time) > 0:
            current_time = visible_time[0]
            minutes = int(current_time // 60)
            seconds = int(current_time % 60)
            self.statusBar().showMessage(f"Posisi: {minutes}:{seconds:02d}")
    
    def update_plot(self):
        # Menambah indeks untuk efek scrolling
        if self.signal is None or len(self.signal) == 0:
            return
            
        # Increment indeks berdasarkan kecepatan pemutaran
        base_increment = max(1, int(self.window_size * 0.1))
        
        # Gunakan play_speed sebagai faktor untuk menentukan increment
        adjusted_increment = int(base_increment * self.play_speed)
        
        self.current_index += adjusted_increment
        
        # Jika mencapai akhir sinyal, kembali ke awal
        if self.current_index >= len(self.signal) - self.window_size:
            self.current_index = 0
            self.statusBar().showMessage("Mencapai akhir rekaman, memutar dari awal.")
        
        # Update plot
        self.update_plot_single()
    
    # Untuk memastikan resources dibersihkan saat aplikasi ditutup
    def closeEvent(self, event):
        if self.timer.isActive():
            self.timer.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Set style untuk tampilan yang lebih modern
    app.setStyle("Fusion")
    
    window = ECGMonitor()
    window.show()
    sys.exit(app.exec_())
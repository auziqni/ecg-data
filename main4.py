import sys
import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QWidget, 
                            QPushButton, QHBoxLayout, QLabel, QComboBox, 
                            QCheckBox, QSlider, QSpinBox, QGridLayout, QGroupBox,
                            QScrollArea, QSizePolicy, QTextEdit, QFileDialog,
                            QMessageBox)
from PyQt5.QtCore import QTimer, Qt
import wfdb
import os
import struct
from datetime import datetime

class ECGToBinaryConverter(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Parameter plot
        self.window_size = 2000
        self.sample_rate = 360
        self.current_index = 0
        self.playing = False
        self.play_speed = 1.0
        
        # Standard 12-lead ECG channel names
        self.standard_channels = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 
                                 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        self.max_channels = 12
        
        # Data
        self.record_name = None
        self.record = None
        self.signal = None
        self.time = None
        self.show_channel = [True] * self.max_channels
        self.channel_colors = [
            (255, 0, 0), (0, 0, 255), (0, 128, 0), (128, 0, 128),
            (255, 165, 0), (0, 128, 128), (128, 0, 0), (0, 0, 128),
            (128, 128, 0), (255, 0, 255), (0, 0, 0), (64, 64, 64)
        ]
        
        # Folder paths
        self.sample_folder = "sample"
        self.output_folder = "hasil"
        
        # Create output folder if not exists
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        
        # Window setup
        self.setWindowTitle("ECG PhysioNet to Binary Converter")
        self.setGeometry(100, 50, 1400, 900)
        
        # Main widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        
        # Create UI components
        self.create_control_panel()
        self.create_plot_widgets()
        self.create_info_panel()
        
        # Timer for animation
        self.timer = QTimer()
        self.timer.setInterval(25)
        self.timer.timeout.connect(self.update_plot)
        
        # Status bar
        self.statusBar().showMessage("Ready. Please select a record from the sample folder.")
        
        # Load available records
        self.load_available_records()
    
    def create_control_panel(self):
        # Control panel
        self.control_group = QGroupBox("Controls")
        self.main_layout.addWidget(self.control_group)
        self.control_layout = QGridLayout(self.control_group)
        
        # Record selection
        self.record_label = QLabel("Select Record:")
        self.control_layout.addWidget(self.record_label, 0, 0)
        
        self.record_combo = QComboBox()
        self.record_combo.currentTextChanged.connect(self.load_record)
        self.control_layout.addWidget(self.record_combo, 0, 1, 1, 2)
        
        # Convert button
        self.convert_button = QPushButton("Convert to Binary")
        self.convert_button.clicked.connect(self.convert_to_binary)
        self.convert_button.setEnabled(False)
        self.control_layout.addWidget(self.convert_button, 0, 3)
        
        # Play controls
        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self.toggle_play)
        self.control_layout.addWidget(self.play_button, 1, 0)
        
        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self.reset_playback)
        self.control_layout.addWidget(self.reset_button, 1, 1)
        
        # Speed control
        self.speed_label = QLabel("Speed:")
        self.control_layout.addWidget(self.speed_label, 1, 2)
        
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setMinimum(1)
        self.speed_slider.setMaximum(50)
        self.speed_slider.setValue(10)
        self.speed_slider.valueChanged.connect(self.change_speed)
        self.control_layout.addWidget(self.speed_slider, 1, 3, 1, 2)
        
        self.speed_value = QLabel("1.0x")
        self.control_layout.addWidget(self.speed_value, 1, 5)
        
        # Window size control
        self.window_label = QLabel("Window Size:")
        self.control_layout.addWidget(self.window_label, 2, 0)
        
        self.window_spinbox = QSpinBox()
        self.window_spinbox.setMinimum(500)
        self.window_spinbox.setMaximum(10000)
        self.window_spinbox.setSingleStep(500)
        self.window_spinbox.setValue(2000)
        self.window_spinbox.valueChanged.connect(self.change_window_size)
        self.control_layout.addWidget(self.window_spinbox, 2, 1)
        
        # Info labels
        self.sample_rate_label = QLabel("Sample Rate: -")
        self.control_layout.addWidget(self.sample_rate_label, 2, 2)
        
        self.duration_label = QLabel("Duration: -")
        self.control_layout.addWidget(self.duration_label, 2, 3, 1, 2)
    
    def create_plot_widgets(self):
        # Scroll area for plots
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.main_layout.addWidget(self.scroll_area)
        
        # Container for plots
        self.plots_container = QWidget()
        self.plots_layout = QVBoxLayout(self.plots_container)
        self.scroll_area.setWidget(self.plots_container)
        
        # Create plot widgets
        self.plot_widgets = []
        self.plot_lines = []
        
        for i in range(self.max_channels):
            # Create plot widget
            plot_widget = pg.PlotWidget()
            plot_widget.setBackground('w')
            plot_widget.showGrid(x=True, y=True, alpha=0.3)
            plot_widget.setLabel('left', self.standard_channels[i])
            
            if i == self.max_channels - 1:
                plot_widget.setLabel('bottom', 'Time (seconds)')
            else:
                plot_widget.getAxis('bottom').setStyle(showValues=False)
            
            plot_widget.setTitle(f'{self.standard_channels[i]}')
            plot_widget.setMinimumHeight(100)
            plot_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            
            # Plot line
            pen = pg.mkPen(color=self.channel_colors[i], width=2)
            plot_line = plot_widget.plot(pen=pen)
            
            self.plots_layout.addWidget(plot_widget)
            self.plot_widgets.append(plot_widget)
            self.plot_lines.append(plot_line)
            
            plot_widget.setVisible(False)
        
        # Link X axes
        for i in range(1, self.max_channels):
            self.plot_widgets[i].setXLink(self.plot_widgets[0])
    
    def create_info_panel(self):
        # Information panel
        self.info_group = QGroupBox("Conversion Information")
        self.main_layout.addWidget(self.info_group)
        self.info_layout = QVBoxLayout(self.info_group)
        
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setMaximumHeight(150)
        self.info_layout.addWidget(self.info_text)
    
    def load_available_records(self):
        """Load available records from sample folder"""
        try:
            if not os.path.exists(self.sample_folder):
                os.makedirs(self.sample_folder)
                self.statusBar().showMessage("Sample folder created. Please add PhysioNet files.")
                return
            
            # Find all .hea files in sample folder
            records = []
            for file in os.listdir(self.sample_folder):
                if file.endswith('.hea'):
                    record_name = file[:-4]  # Remove .hea extension
                    records.append(record_name)
            
            if records:
                self.record_combo.addItems(sorted(records))
            else:
                self.statusBar().showMessage("No records found in sample folder.")
        except Exception as e:
            self.statusBar().showMessage(f"Error loading records: {str(e)}")
    
    def load_record(self, record_name):
        """Load selected record"""
        if not record_name:
            return
            
        try:
            self.statusBar().showMessage(f"Loading record {record_name}...")
            
            # Load record from sample folder
            record_path = os.path.join(self.sample_folder, record_name)
            self.record = wfdb.rdrecord(record_path)
            self.signal = self.record.p_signal
            self.sample_rate = self.record.fs
            
            # Create time array
            self.time = np.arange(len(self.signal)) / self.sample_rate
            
            # Update UI
            self.current_index = 0
            self.update_record_info()
            self.update_plots()
            self.convert_button.setEnabled(True)
            
            self.statusBar().showMessage(f"Record {record_name} loaded successfully.")
            
        except Exception as e:
            self.statusBar().showMessage(f"Error loading record: {str(e)}")
            self.convert_button.setEnabled(False)
    
    def update_record_info(self):
        """Update record information display"""
        if self.record is None:
            return
        
        # Update sample rate
        self.sample_rate_label.setText(f"Sample Rate: {self.sample_rate} Hz")
        
        # Update duration
        total_duration = len(self.signal) / self.sample_rate
        minutes = int(total_duration // 60)
        seconds = int(total_duration % 60)
        self.duration_label.setText(f"Duration: {minutes}:{seconds:02d}")
        
        # Update info text
        info_text = f"Record: {self.record_combo.currentText()}\n"
        info_text += f"Channels available: {self.signal.shape[1] if len(self.signal.shape) > 1 else 1}\n"
        info_text += f"Sample rate: {self.sample_rate} Hz\n"
        info_text += f"Total samples: {len(self.signal)}\n"
        info_text += f"Channel names: {', '.join(self.record.sig_name)}\n"
        
        self.info_text.setText(info_text)
        
        # Show/hide plots based on available channels
        for i in range(self.max_channels):
            self.plot_widgets[i].setVisible(True)
    
    def map_channels_to_standard(self):
        """Map available channels to standard 12-lead positions"""
        # Create 12-channel array filled with zeros
        mapped_signal = np.zeros((len(self.signal), 12))
        
        if self.record is None or self.signal is None:
            return mapped_signal
        
        # Get available channel names
        available_channels = self.record.sig_name
        
        # Map each available channel to its standard position
        for i, ch_name in enumerate(available_channels):
            # Normalize channel name (remove spaces, convert to uppercase)
            ch_normalized = ch_name.strip().upper()
            
            # Try to find matching standard channel
            for j, std_ch in enumerate(self.standard_channels):
                std_normalized = std_ch.upper()
                
                # Handle different naming conventions
                if (ch_normalized == std_normalized or 
                    ch_normalized == std_normalized.replace('A', '') or  # aVR -> VR
                    (ch_normalized == 'MLII' and std_normalized == 'II') or  # MLII -> II
                    (ch_normalized == 'MLI' and std_normalized == 'I')):  # MLI -> I
                    
                    if len(self.signal.shape) > 1:
                        mapped_signal[:, j] = self.signal[:, i]
                    else:
                        mapped_signal[:, j] = self.signal[:]
                    break
        
        return mapped_signal
    
    def convert_to_binary(self):
        """Convert ECG data to binary format"""
        if self.signal is None:
            return
        
        try:
            # Map channels to standard 12-lead positions
            mapped_signal = self.map_channels_to_standard()
            
            # Find min/max values for scaling
            min_val = np.min(mapped_signal)
            max_val = np.max(mapped_signal)
            
            # Convert to 12-bit unsigned (0-4095)
            # Scale and shift to fit in 12-bit range
            scaled_signal = (mapped_signal - min_val) / (max_val - min_val) * 4095
            scaled_signal = np.round(scaled_signal).astype(np.uint16)
            
            # Create output filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"{self.record_combo.currentText()}_{timestamp}.bin"
            output_path = os.path.join(self.output_folder, output_filename)
            
            # Write binary file
            with open(output_path, 'wb') as f:
                for sample_idx in range(len(scaled_signal)):
                    for channel_idx in range(12):
                        # Get value (already uint16)
                        value = scaled_signal[sample_idx, channel_idx]
                        # Ensure it's within 12-bit range
                        value = min(value, 4095)
                        # Write as little-endian uint16
                        f.write(struct.pack('<H', value))
            
            # Validate the written file
            file_size = os.path.getsize(output_path)
            expected_size = len(scaled_signal) * 12 * 2  # samples * channels * bytes_per_value
            
            # Update info display
            info_text = self.info_text.toPlainText()
            info_text += f"\n\n--- Conversion Results ---\n"
            info_text += f"Output file: {output_filename}\n"
            info_text += f"File size: {file_size} bytes\n"
            info_text += f"Expected size: {expected_size} bytes\n"
            info_text += f"Scaling: [{min_val:.3f}, {max_val:.3f}] -> [0, 4095]\n"
            info_text += f"Status: {'Success' if file_size == expected_size else 'Size mismatch!'}\n"
            
            self.info_text.setText(info_text)
            
            # Show success message
            if file_size == expected_size:
                QMessageBox.information(self, "Success", 
                    f"Binary file created successfully!\n\nFile: {output_filename}\n"
                    f"Size: {file_size} bytes\n"
                    f"Location: {self.output_folder}")
            else:
                QMessageBox.warning(self, "Warning", 
                    f"File created but size mismatch detected!\n"
                    f"Expected: {expected_size} bytes\n"
                    f"Actual: {file_size} bytes")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Conversion failed: {str(e)}")
    
    def update_plots(self):
        """Update all plots with current window"""
        if self.signal is None:
            return
        
        # Map channels for display
        mapped_signal = self.map_channels_to_standard()
        
        end_idx = min(self.current_index + self.window_size, len(self.signal))
        visible_time = self.time[self.current_index:end_idx]
        
        # Update each channel
        for i in range(self.max_channels):
            visible_data = mapped_signal[self.current_index:end_idx, i]
            
            # Update plot
            self.plot_lines[i].setData(visible_time, visible_data)
            
            # Auto-scale
            if len(visible_data) > 0 and np.any(visible_data != 0):
                min_val = np.min(visible_data)
                max_val = np.max(visible_data)
                padding = (max_val - min_val) * 0.1
                self.plot_widgets[i].setYRange(min_val - padding, max_val + padding)
        
        # Update x range
        if len(visible_time) > 0:
            self.plot_widgets[0].setXRange(visible_time[0], visible_time[-1])
    
    def toggle_play(self):
        """Toggle play/pause"""
        if self.playing:
            self.timer.stop()
            self.playing = False
            self.play_button.setText("Play")
        else:
            self.timer.start()
            self.playing = True
            self.play_button.setText("Pause")
    
    def reset_playback(self):
        """Reset to beginning"""
        self.current_index = 0
        self.update_plots()
    
    def change_speed(self, value):
        """Change playback speed"""
        self.play_speed = value / 10.0
        self.speed_value.setText(f"{self.play_speed:.1f}x")
    
    def change_window_size(self, value):
        """Change display window size"""
        self.window_size = value
        self.update_plots()
    
    def update_plot(self):
        """Timer update for animation"""
        if self.signal is None:
            return
        
        # Increment based on speed
        increment = int(self.window_size * 0.1 * self.play_speed)
        self.current_index += increment
        
        # Loop back to start
        if self.current_index >= len(self.signal) - self.window_size:
            self.current_index = 0
        
        self.update_plots()
    
    def closeEvent(self, event):
        """Clean up on close"""
        if self.timer.isActive():
            self.timer.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    window = ECGToBinaryConverter()
    window.show()
    sys.exit(app.exec_())
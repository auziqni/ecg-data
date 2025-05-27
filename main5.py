import sys
import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QWidget, 
                            QPushButton, QHBoxLayout, QLabel, QComboBox, 
                            QCheckBox, QSlider, QSpinBox, QGridLayout, QGroupBox,
                            QScrollArea, QSizePolicy, QTextEdit, QFileDialog,
                            QMessageBox, QDoubleSpinBox, QFrame)
from PyQt5.QtCore import QTimer, Qt, QPropertyAnimation, QRect, pyqtSignal
from PyQt5.QtGui import QIcon, QPainter, QPolygon
import wfdb
import os
import struct
from datetime import datetime

class CollapsibleInfoBar(QWidget):
    """Collapsible information bar with animation"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.is_collapsed = True
        self.animation_duration = 200
        
        # Main layout
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)
        
        # Header bar with arrow
        self.header = QFrame()
        self.header.setFrameStyle(QFrame.Box)
        self.header.setStyleSheet("""
            QFrame {
                background-color: #f0f0f0;
                border: 1px solid #ccc;
            }
        """)
        self.header.setFixedHeight(30)
        self.header.mousePressEvent = self.toggle_collapse
        self.header.setCursor(Qt.PointingHandCursor)
        
        self.header_layout = QHBoxLayout(self.header)
        self.arrow_label = QLabel("▶")  # Right arrow when collapsed
        self.arrow_label.setStyleSheet("font-size: 12px;")
        self.title_label = QLabel("Conversion Information")
        self.title_label.setStyleSheet("font-weight: bold;")
        
        self.header_layout.addWidget(self.arrow_label)
        self.header_layout.addWidget(self.title_label)
        self.header_layout.addStretch()
        
        # Content area
        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout(self.content_widget)
        
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.content_layout.addWidget(self.info_text)
        
        # Add to main layout
        self.main_layout.addWidget(self.header)
        self.main_layout.addWidget(self.content_widget)
        
        # Initially hide content
        self.content_widget.setMaximumHeight(0)
        self.content_widget.setMinimumHeight(0)
        
        # Animation
        self.animation = QPropertyAnimation(self.content_widget, b"maximumHeight")
        self.animation.setDuration(self.animation_duration)
    
    def toggle_collapse(self, event):
        """Toggle collapsed state with animation"""
        self.is_collapsed = not self.is_collapsed
        
        if self.is_collapsed:
            self.arrow_label.setText("▶")
            self.animation.setStartValue(self.content_widget.height())
            self.animation.setEndValue(0)
        else:
            self.arrow_label.setText("▼")
            self.animation.setStartValue(self.content_widget.height())
            # Set to parent height / 2
            parent_height = self.parent().height() if self.parent() else 400
            self.animation.setEndValue(parent_height // 2)
        
        self.animation.start()
    
    def set_text(self, text):
        """Set info text"""
        self.info_text.setText(text)

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
        self.signal_trimmed = None  # Trimmed signal for display/conversion
        self.time = None
        self.time_trimmed = None
        self.show_channel = [True] * self.max_channels
        self.channel_colors = [
            (255, 0, 0), (0, 0, 255), (0, 128, 0), (128, 0, 128),
            (255, 165, 0), (0, 128, 128), (128, 0, 0), (0, 0, 128),
            (128, 128, 0), (255, 0, 255), (0, 0, 0), (64, 64, 64)
        ]
        
        # Trim parameters
        self.trim_start = 0.0
        self.trim_end = 0.0
        
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
        self.create_channel_controls()
        self.create_plot_area()
        self.create_info_bar()
        
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
        
        # Row 0: Record selection and convert button
        self.record_label = QLabel("Select Record:")
        self.control_layout.addWidget(self.record_label, 0, 0)
        
        self.record_combo = QComboBox()
        self.record_combo.currentTextChanged.connect(self.load_record)
        self.control_layout.addWidget(self.record_combo, 0, 1, 1, 2)
        
        self.convert_button = QPushButton("Convert to Binary")
        self.convert_button.clicked.connect(self.convert_to_binary)
        self.convert_button.setEnabled(False)
        self.control_layout.addWidget(self.convert_button, 0, 3)
        
        # Row 1: Play controls and speed
        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self.toggle_play)
        self.control_layout.addWidget(self.play_button, 1, 0)
        
        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self.reset_playback)
        self.control_layout.addWidget(self.reset_button, 1, 1)
        
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
        
        # Row 2: Window size and info
        self.window_label = QLabel("Window Size:")
        self.control_layout.addWidget(self.window_label, 2, 0)
        
        self.window_spinbox = QSpinBox()
        self.window_spinbox.setMinimum(500)
        self.window_spinbox.setMaximum(10000)
        self.window_spinbox.setSingleStep(500)
        self.window_spinbox.setValue(2000)
        self.window_spinbox.valueChanged.connect(self.change_window_size)
        self.control_layout.addWidget(self.window_spinbox, 2, 1)
        
        self.sample_rate_label = QLabel("Sample Rate: -")
        self.control_layout.addWidget(self.sample_rate_label, 2, 2)
        
        self.duration_label = QLabel("Duration: -")
        self.control_layout.addWidget(self.duration_label, 2, 3)
        
        self.total_samples_label = QLabel("Total: - samples")
        self.control_layout.addWidget(self.total_samples_label, 2, 4, 1, 2)
        
        # Row 3: Trim controls
        self.trim_label = QLabel("Trim Signal:")
        self.control_layout.addWidget(self.trim_label, 3, 0)
        
        self.start_label = QLabel("Start (s):")
        self.control_layout.addWidget(self.start_label, 3, 1)
        
        self.start_spinbox = QDoubleSpinBox()
        self.start_spinbox.setMinimum(0.0)
        self.start_spinbox.setMaximum(9999.0)
        self.start_spinbox.setDecimals(2)
        self.start_spinbox.setSingleStep(0.1)
        self.start_spinbox.valueChanged.connect(self.update_trim)
        self.control_layout.addWidget(self.start_spinbox, 3, 2)
        
        self.end_label = QLabel("End (s):")
        self.control_layout.addWidget(self.end_label, 3, 3)
        
        self.end_spinbox = QDoubleSpinBox()
        self.end_spinbox.setMinimum(0.0)
        self.end_spinbox.setMaximum(9999.0)
        self.end_spinbox.setDecimals(2)
        self.end_spinbox.setSingleStep(0.1)
        self.end_spinbox.valueChanged.connect(self.update_trim)
        self.control_layout.addWidget(self.end_spinbox, 3, 4)
        
        self.trim_info_label = QLabel("Trimmed: - samples")
        self.control_layout.addWidget(self.trim_info_label, 3, 5)
    
    def create_channel_controls(self):
        # Channel visibility controls
        self.channel_group = QGroupBox("Channel Visibility")
        self.main_layout.addWidget(self.channel_group)
        self.channel_layout = QGridLayout(self.channel_group)
        
        self.channel_checkboxes = []
        for i in range(self.max_channels):
            checkbox = QCheckBox(self.standard_channels[i])
            checkbox.setChecked(True)
            checkbox.stateChanged.connect(lambda state, idx=i: self.toggle_channel(idx, state))
            
            row = i // 6
            col = i % 6
            self.channel_layout.addWidget(checkbox, row, col)
            self.channel_checkboxes.append(checkbox)
    
    def create_plot_area(self):
        # Container for plot area
        self.plot_container = QWidget()
        self.plot_container_layout = QVBoxLayout(self.plot_container)
        self.plot_container_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.addWidget(self.plot_container)
        
        # Scroll area for plots
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.plot_container_layout.addWidget(self.scroll_area)
        
        # Container for plots
        self.plots_widget = QWidget()
        self.plots_layout = QVBoxLayout(self.plots_widget)
        self.plots_layout.setSpacing(2)
        self.scroll_area.setWidget(self.plots_widget)
        
        # Message label for no channels
        self.no_channel_label = QLabel("No channels selected")
        self.no_channel_label.setAlignment(Qt.AlignCenter)
        self.no_channel_label.setStyleSheet("""
            QLabel {
                font-size: 16px;
                color: #666;
                padding: 50px;
            }
        """)
        self.no_channel_label.hide()
        self.plots_layout.addWidget(self.no_channel_label)
        
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
    
    def create_info_bar(self):
        # Create collapsible info bar
        self.info_bar = CollapsibleInfoBar(self.plot_container)
        self.info_bar.setStyleSheet("""
            QWidget {
                background-color: white;
                border: 1px solid #ccc;
            }
        """)
        
        # Position at bottom of plot container
        self.info_bar.move(0, self.plot_container.height() - 30)
        self.info_bar.resize(self.plot_container.width(), 30)
    
    def resizeEvent(self, event):
        """Handle window resize to reposition info bar"""
        super().resizeEvent(event)
        if hasattr(self, 'info_bar') and hasattr(self, 'plot_container'):
            self.info_bar.move(0, self.plot_container.height() - 30)
            self.info_bar.resize(self.plot_container.width(), 30)
    
    def load_available_records(self):
        """Load available records from sample folder"""
        try:
            if not os.path.exists(self.sample_folder):
                os.makedirs(self.sample_folder)
                self.statusBar().showMessage("Sample folder created. Please add PhysioNet files.")
                return
            
            records = []
            for file in os.listdir(self.sample_folder):
                if file.endswith('.hea'):
                    record_name = file[:-4]
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
            
            # Load record
            record_path = os.path.join(self.sample_folder, record_name)
            self.record = wfdb.rdrecord(record_path)
            self.signal = self.record.p_signal
            self.sample_rate = self.record.fs
            
            # Create time array
            self.time = np.arange(len(self.signal)) / self.sample_rate
            
            # Reset trim parameters
            self.trim_start = 0.0
            self.trim_end = len(self.signal) / self.sample_rate
            
            # Update trim spinboxes
            self.start_spinbox.setMaximum(self.trim_end)
            self.end_spinbox.setMaximum(self.trim_end)
            self.end_spinbox.setValue(self.trim_end)
            
            # Apply initial trim
            self.update_trim()
            
            # Update UI
            self.current_index = 0
            self.update_record_info()
            self.update_plots()
            self.convert_button.setEnabled(True)
            
            self.statusBar().showMessage(f"Record {record_name} loaded successfully.")
            
        except Exception as e:
            self.statusBar().showMessage(f"Error loading record: {str(e)}")
            self.convert_button.setEnabled(False)
    
    def update_trim(self):
        """Update signal trimming"""
        if self.signal is None:
            return
        
        # Get trim values
        self.trim_start = self.start_spinbox.value()
        self.trim_end = self.end_spinbox.value()
        
        # Ensure valid range
        if self.trim_end <= self.trim_start:
            self.trim_end = self.trim_start + 0.1
            self.end_spinbox.setValue(self.trim_end)
        
        # Calculate sample indices
        start_idx = int(self.trim_start * self.sample_rate)
        end_idx = int(self.trim_end * self.sample_rate)
        
        # Trim signal and time
        self.signal_trimmed = self.signal[start_idx:end_idx]
        self.time_trimmed = self.time[start_idx:end_idx] - self.time[start_idx]
        
        # Update info
        trimmed_samples = len(self.signal_trimmed)
        self.trim_info_label.setText(f"Trimmed: {trimmed_samples:,} samples")
        
        # Reset playback and update plots
        self.current_index = 0
        self.update_plots()
    
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
        
        # Update total samples
        self.total_samples_label.setText(f"Total: {len(self.signal):,} samples")
        
        # Update info text
        info_text = f"Record: {self.record_combo.currentText()}\n"
        info_text += f"Channels available: {self.signal.shape[1] if len(self.signal.shape) > 1 else 1}\n"
        info_text += f"Sample rate: {self.sample_rate} Hz\n"
        info_text += f"Total samples: {len(self.signal):,}\n"
        info_text += f"Channel names: {', '.join(self.record.sig_name)}\n"
        
        self.info_bar.set_text(info_text)
    
    def toggle_channel(self, channel, state):
        """Toggle channel visibility"""
        self.show_channel[channel] = state > 0
        self.update_plot_layout()
        self.update_plots()
    
    def update_plot_layout(self):
        """Update plot layout based on visible channels"""
        # Count visible channels
        visible_count = sum(self.show_channel)
        
        if visible_count == 0:
            # Hide all plots and show message
            for plot in self.plot_widgets:
                plot.hide()
            self.no_channel_label.show()
            return
        else:
            self.no_channel_label.hide()
        
        # Calculate height for each plot
        available_height = self.scroll_area.height()
        if visible_count <= 3:
            plot_height = available_height // visible_count
        else:
            plot_height = available_height // 3  # Minimum 1/3 height
        
        # Update plot visibility and height
        for i, plot in enumerate(self.plot_widgets):
            if self.show_channel[i]:
                plot.setMinimumHeight(plot_height)
                plot.setMaximumHeight(plot_height if visible_count <= 3 else 16777215)
                plot.show()
            else:
                plot.hide()
    
    def map_channels_to_standard(self):
        """Map available channels to standard 12-lead positions"""
        # Use trimmed signal if available
        signal_to_map = self.signal_trimmed if self.signal_trimmed is not None else self.signal
        
        # Create 12-channel array filled with zeros
        mapped_signal = np.zeros((len(signal_to_map), 12))
        
        if self.record is None or signal_to_map is None:
            return mapped_signal
        
        # Get available channel names
        available_channels = self.record.sig_name
        
        # Map each available channel to its standard position
        for i, ch_name in enumerate(available_channels):
            ch_normalized = ch_name.strip().upper()
            
            for j, std_ch in enumerate(self.standard_channels):
                std_normalized = std_ch.upper()
                
                if (ch_normalized == std_normalized or 
                    ch_normalized == std_normalized.replace('A', '') or
                    (ch_normalized == 'MLII' and std_normalized == 'II') or
                    (ch_normalized == 'MLI' and std_normalized == 'I')):
                    
                    if len(signal_to_map.shape) > 1:
                        mapped_signal[:, j] = signal_to_map[:, i]
                    else:
                        mapped_signal[:, j] = signal_to_map[:]
                    break
        
        return mapped_signal
    
    def convert_to_binary(self):
        """Convert ECG data to binary format"""
        if self.signal_trimmed is None:
            return
        
        try:
            # Map channels to standard 12-lead positions
            mapped_signal = self.map_channels_to_standard()
            
            # Find min/max values for scaling
            min_val = np.min(mapped_signal)
            max_val = np.max(mapped_signal)
            
            # Convert to 12-bit unsigned (0-4095)
            scaled_signal = (mapped_signal - min_val) / (max_val - min_val) * 4095
            scaled_signal = np.round(scaled_signal).astype(np.uint16)
            
            # Create output filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"{self.record_combo.currentText()}_{timestamp}.bin"
            output_path = os.path.join(self.output_folder, output_filename)
            
            # Write binary file
            with open(output_path, 'wb') as f:
                for sample_idx in range(len(scaled_signal)):
                    for channel_idx in range(12):
                        value = scaled_signal[sample_idx, channel_idx]
                        value = min(value, 4095)
                        f.write(struct.pack('<H', value))
            
            # Validate
            file_size = os.path.getsize(output_path)
            expected_size = len(scaled_signal) * 12 * 2
            
            # Update info display
            info_text = self.info_bar.info_text.toPlainText()
            info_text += f"\n\n--- Conversion Results ---\n"
            info_text += f"Output file: {output_filename}\n"
            info_text += f"Trimmed duration: {self.trim_start:.2f}s - {self.trim_end:.2f}s\n"
            info_text += f"Samples converted: {len(scaled_signal):,}\n"
            info_text += f"File size: {file_size:,} bytes\n"
            info_text += f"Expected size: {expected_size:,} bytes\n"
            info_text += f"Scaling: [{min_val:.3f}, {max_val:.3f}] -> [0, 4095]\n"
            info_text += f"Status: {'Success' if file_size == expected_size else 'Size mismatch!'}\n"
            
            self.info_bar.set_text(info_text)
            
            # Show success message
            if file_size == expected_size:
                QMessageBox.information(self, "Success", 
                    f"Binary file created successfully!\n\nFile: {output_filename}\n"
                    f"Size: {file_size:,} bytes\n"
                    f"Duration: {self.trim_end - self.trim_start:.2f}s\n"
                    f"Location: {self.output_folder}")
            else:
                QMessageBox.warning(self, "Warning", 
                    f"File created but size mismatch detected!")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Conversion failed: {str(e)}")
    
    def update_plots(self):
        """Update all plots with current window"""
        if self.signal_trimmed is None:
            return
        
        # Map channels for display
        mapped_signal = self.map_channels_to_standard()
        
        end_idx = min(self.current_index + self.window_size, len(self.signal_trimmed))
        visible_time = self.time_trimmed[self.current_index:end_idx]
        
        # Update each channel
        for i in range(self.max_channels):
            if self.show_channel[i]:
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
        if self.signal_trimmed is None:
            return
        
        # Increment based on speed
        increment = int(self.window_size * 0.1 * self.play_speed)
        self.current_index += increment
        
        # Loop back to start
        if self.current_index >= len(self.signal_trimmed) - self.window_size:
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
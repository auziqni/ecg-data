import sys
import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QWidget, 
                            QPushButton, QHBoxLayout, QLabel, QComboBox, 
                            QCheckBox, QSlider, QSpinBox, QGridLayout, QGroupBox,
                            QScrollArea, QSizePolicy, QTextEdit, QFileDialog,
                            QMessageBox, QDoubleSpinBox, QFrame, QListWidget,
                            QListWidgetItem, QSplitter)
from PyQt5.QtCore import QTimer, Qt, QPropertyAnimation, QRect, pyqtSignal, QDateTime
from PyQt5.QtGui import QIcon, QPainter, QPolygon, QFont
import wfdb
import os
import struct
from datetime import datetime
from collections import deque

class ConversionHistoryItem:
    """Store conversion history data"""
    def __init__(self, filename, timestamp, duration, samples, file_size, status):
        self.filename = filename
        self.timestamp = timestamp
        self.duration = duration
        self.samples = samples
        self.file_size = file_size
        self.status = status

class SidebarInfoPanel(QFrame):
    """Collapsible sidebar panel for conversion information"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.is_collapsed = True
        self.animation_duration = 200
        self.panel_width = 400  # 1/3 of typical 1200px width
        
        # Style
        self.setFrameStyle(QFrame.Box)
        self.setStyleSheet("""
            QFrame {
                background-color: #f8f8f8;
                border: 1px solid #ccc;
                border-left: none;
            }
        """)
        
        # Main layout
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)
        
        # Header
        self.header = QFrame()
        self.header.setFixedHeight(40)
        self.header.setStyleSheet("""
            QFrame {
                background-color: #e0e0e0;
                border-bottom: 1px solid #ccc;
            }
        """)
        
        self.header_layout = QHBoxLayout(self.header)
        self.title_label = QLabel("Conversion Information")
        self.title_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.header_layout.addWidget(self.title_label)
        self.header_layout.addStretch()
        
        # Content area with splitter
        self.content_splitter = QSplitter(Qt.Vertical)
        
        # Current conversion info
        self.current_group = QGroupBox("Current Conversion")
        self.current_layout = QVBoxLayout(self.current_group)
        self.current_info = QTextEdit()
        self.current_info.setReadOnly(True)
        self.current_layout.addWidget(self.current_info)
        
        # History
        self.history_group = QGroupBox("Conversion History (Last 10)")
        self.history_layout = QVBoxLayout(self.history_group)
        
        # History list
        self.history_list = QListWidget()
        self.history_list.setAlternatingRowColors(True)
        self.history_layout.addWidget(self.history_list)
        
        # Clear history button
        self.clear_history_btn = QPushButton("Clear History")
        self.clear_history_btn.clicked.connect(self.clear_history)
        self.history_layout.addWidget(self.clear_history_btn)
        
        # Add to splitter
        self.content_splitter.addWidget(self.current_group)
        self.content_splitter.addWidget(self.history_group)
        self.content_splitter.setSizes([200, 400])
        
        # Add to main layout
        self.main_layout.addWidget(self.header)
        self.main_layout.addWidget(self.content_splitter)
        
        # Animation
        self.animation = QPropertyAnimation(self, b"geometry")
        self.animation.setDuration(self.animation_duration)
        
        # History storage (max 10 items)
        self.history_items = deque(maxlen=10)
        
        # Initially collapsed
        self.setFixedWidth(0)
    
    def toggle(self):
        """Toggle sidebar visibility"""
        self.is_collapsed = not self.is_collapsed
        
        parent_rect = self.parent().rect()
        
        if self.is_collapsed:
            start_rect = QRect(0, 0, self.panel_width, parent_rect.height())
            end_rect = QRect(-self.panel_width, 0, self.panel_width, parent_rect.height())
        else:
            start_rect = QRect(-self.panel_width, 0, self.panel_width, parent_rect.height())
            end_rect = QRect(0, 0, self.panel_width, parent_rect.height())
            self.raise_()  # Bring to front
        
        self.setFixedWidth(self.panel_width)
        self.animation.setStartValue(start_rect)
        self.animation.setEndValue(end_rect)
        self.animation.start()
    
    def open(self):
        """Open sidebar if closed"""
        if self.is_collapsed:
            self.toggle()
    
    def set_current_info(self, text):
        """Update current conversion info"""
        self.current_info.setText(text)
    
    def add_history_item(self, item):
        """Add item to history"""
        self.history_items.append(item)
        self.update_history_display()
    
    def update_history_display(self):
        """Update history list display"""
        self.history_list.clear()
        
        for item in reversed(self.history_items):  # Show newest first
            # Create list item with formatted text
            text = f"{item.timestamp} - {item.filename}\n"
            text += f"Duration: {item.duration:.2f}s, Samples: {item.samples:,}\n"
            text += f"Size: {item.file_size:,} bytes - {item.status}"
            
            list_item = QListWidgetItem(text)
            if item.status != "Success":
                list_item.setForeground(Qt.red)
            
            self.history_list.addItem(list_item)
    
    def clear_history(self):
        """Clear conversion history"""
        self.history_items.clear()
        self.history_list.clear()

class ECGToBinaryConverter(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Parameter plot
        self.window_size = 2000
        self.sample_rate = 360
        self.current_index = 0
        self.playing = False
        self.play_speed = 1.0
        
        # ESP32 ADC Configuration
        self.adc_resolution = 4095  # 12-bit ADC (0-4095)
        self.vcc = 3.3  # ESP32 VCC voltage
        self.gain = 300  # Signal gain
        self.offset_voltage = 1.65  # Offset voltage (VCC/2)
        self.offset_adc = 2048  # Offset in ADC counts (4095/2)
        
        # Y-axis display modes
        self.y_axis_modes = [
            "Asli (mV)",
            "Hasil (12bit)", 
            "Tegangan Hasil (V)"
        ]
        self.current_y_mode = 0
        
        # Standard 12-lead ECG channel names
        self.standard_channels = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 
                                 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        self.max_channels = 12
        
        # Data
        self.record_name = None
        self.record = None
        self.signal = None
        self.signal_trimmed = None
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
        self.time_offset = 0.0  # For x-axis display
        
        # Warning tracking
        self.clipping_warnings = []
        
        # Folder paths
        self.sample_folder = "sample"
        self.output_folder = "hasil"
        
        # Create output folder if not exists
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        
        # Window setup
        self.setWindowTitle("ECG PhysioNet to Binary Converter - ESP32 Compatible")
        self.setGeometry(100, 50, 1400, 900)
        
        # Main widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        
        # Create UI components
        self.create_control_panel()
        self.create_channel_controls()
        self.create_plot_area()
        
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
        
        # Row 0: Record selection, Y-axis mode, convert button, and info panel toggle
        self.record_label = QLabel("Select Record:")
        self.control_layout.addWidget(self.record_label, 0, 0)
        
        self.record_combo = QComboBox()
        self.record_combo.currentTextChanged.connect(self.load_record)
        self.control_layout.addWidget(self.record_combo, 0, 1)
        
        # Y-axis mode selector
        self.y_mode_label = QLabel("Y-Axis Mode:")
        self.control_layout.addWidget(self.y_mode_label, 0, 2)
        
        self.y_mode_combo = QComboBox()
        self.y_mode_combo.addItems(self.y_axis_modes)
        self.y_mode_combo.currentIndexChanged.connect(self.change_y_mode)
        self.control_layout.addWidget(self.y_mode_combo, 0, 3)
        
        self.convert_button = QPushButton("Convert to Binary")
        self.convert_button.clicked.connect(self.convert_to_binary)
        self.convert_button.setEnabled(False)
        self.control_layout.addWidget(self.convert_button, 0, 4)
        
        # Toggle info panel button
        self.info_toggle_btn = QPushButton("Show/Hide Info Panel")
        self.info_toggle_btn.clicked.connect(self.toggle_info_panel)
        self.control_layout.addWidget(self.info_toggle_btn, 0, 5)
        
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
        
        # Row 2: Window size and ESP32 info
        self.window_label = QLabel("Window Size:")
        self.control_layout.addWidget(self.window_label, 2, 0)
        
        self.window_spinbox = QSpinBox()
        self.window_spinbox.setMinimum(100)
        self.window_spinbox.setMaximum(10000)
        self.window_spinbox.setSingleStep(100)
        self.window_spinbox.setValue(2000)
        self.window_spinbox.valueChanged.connect(self.change_window_size)
        self.control_layout.addWidget(self.window_spinbox, 2, 1)
        
        self.sample_rate_label = QLabel("Sample Rate: -")
        self.control_layout.addWidget(self.sample_rate_label, 2, 2)
        
        self.duration_label = QLabel("Duration: -")
        self.control_layout.addWidget(self.duration_label, 2, 3)
        
        self.esp32_info_label = QLabel("ESP32: Gain=300x, ADC=12bit")
        self.esp32_info_label.setStyleSheet("color: blue; font-weight: bold;")
        self.control_layout.addWidget(self.esp32_info_label, 2, 4, 1, 2)
        
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
        
        # Add Select All / Deselect All buttons
        self.select_all_btn = QPushButton("Select All")
        self.select_all_btn.clicked.connect(self.select_all_channels)
        self.channel_layout.addWidget(self.select_all_btn, 0, 0, 1, 3)
        
        self.deselect_all_btn = QPushButton("Deselect All")
        self.deselect_all_btn.clicked.connect(self.deselect_all_channels)
        self.channel_layout.addWidget(self.deselect_all_btn, 0, 3, 1, 3)
        
        self.channel_checkboxes = []
        for i in range(self.max_channels):
            checkbox = QCheckBox(self.standard_channels[i])
            checkbox.setChecked(True)
            checkbox.stateChanged.connect(lambda state, idx=i: self.toggle_channel(idx, state))
            
            row = (i // 6) + 1  # Start from row 1 due to buttons
            col = i % 6
            self.channel_layout.addWidget(checkbox, row, col)
            self.channel_checkboxes.append(checkbox)
    
    def create_plot_area(self):
        # Container for plot area with relative positioning
        self.plot_container = QWidget()
        self.plot_container_layout = QHBoxLayout(self.plot_container)
        self.plot_container_layout.setContentsMargins(0, 0, 0, 0)
        self.plot_container_layout.setSpacing(0)
        self.main_layout.addWidget(self.plot_container)
        
        # Create sidebar (initially hidden)
        self.info_sidebar = SidebarInfoPanel(self.plot_container)
        
        # Plot area widget
        self.plot_area_widget = QWidget()
        self.plot_area_layout = QVBoxLayout(self.plot_area_widget)
        self.plot_area_layout.setContentsMargins(0, 0, 0, 0)
        self.plot_container_layout.addWidget(self.plot_area_widget)
        
        # Scroll area for plots
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.plot_area_layout.addWidget(self.scroll_area)
        
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
            plot_widget.setLabel('left', self.get_y_label(i))
            
            # Always show x-axis values for all plots, but only label the last one
            plot_widget.getAxis('bottom').setStyle(showValues=True)
            
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
    
    def get_y_label(self, channel_idx):
        """Get Y-axis label based on current mode"""
        channel = self.standard_channels[channel_idx]
        mode = self.y_axis_modes[self.current_y_mode]
        return f"{channel}\n({mode})"
    
    def change_y_mode(self, index):
        """Change Y-axis display mode"""
        self.current_y_mode = index
        
        # Update all plot labels
        for i in range(self.max_channels):
            self.plot_widgets[i].setLabel('left', self.get_y_label(i))
        
        # Update plots with new scaling
        self.update_plots()
    
    def convert_mv_to_adc(self, signal_mv):
        """Convert mV signal to ESP32 ADC values"""
        # Apply gain (300x)
        signal_gained = signal_mv * self.gain / 1000.0  # Convert mV to V and apply gain
        
        # Add offset (1.65V)
        signal_offset = signal_gained + self.offset_voltage
        
        # Convert to ADC counts
        adc_values = signal_offset * self.adc_resolution / self.vcc
        
        # Check for clipping and track warnings
        clipped_low = np.sum(adc_values < 0)
        clipped_high = np.sum(adc_values > self.adc_resolution)
        
        if clipped_low > 0 or clipped_high > 0:
            warning = f"Clipping detected: {clipped_low} samples < 0, {clipped_high} samples > {self.adc_resolution}"
            if warning not in self.clipping_warnings:
                self.clipping_warnings.append(warning)
        
        # Clip values to valid range
        adc_values = np.clip(adc_values, 0, self.adc_resolution)
        
        return adc_values
    
    def convert_adc_to_voltage(self, adc_values):
        """Convert ADC values back to voltage"""
        return adc_values * self.vcc / self.adc_resolution
    
    def get_display_data(self, signal_mv):
        """Get data for display based on current Y-axis mode"""
        if self.current_y_mode == 0:  # Asli (mV)
            return signal_mv
        elif self.current_y_mode == 1:  # Hasil (12bit)
            return self.convert_mv_to_adc(signal_mv)
        else:  # Tegangan Hasil (V)
            adc_values = self.convert_mv_to_adc(signal_mv)
            return self.convert_adc_to_voltage(adc_values)
    
    def resizeEvent(self, event):
        """Handle window resize to reposition sidebar"""
        super().resizeEvent(event)
        if hasattr(self, 'info_sidebar') and hasattr(self, 'plot_container'):
            self.info_sidebar.setGeometry(
                0 if not self.info_sidebar.is_collapsed else -self.info_sidebar.panel_width,
                0,
                self.info_sidebar.panel_width,
                self.plot_container.height()
            )
    
    def toggle_info_panel(self):
        """Toggle info sidebar"""
        self.info_sidebar.toggle()
    
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
            
            # Clear previous warnings
            self.clipping_warnings = []
            
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
            
            # Update sidebar info
            info_text = f"Record: {record_name}\n"
            info_text += f"Channels available: {self.signal.shape[1] if len(self.signal.shape) > 1 else 1}\n"
            info_text += f"Sample rate: {self.sample_rate} Hz\n"
            info_text += f"Total samples: {len(self.signal):,}\n"
            info_text += f"Channel names: {', '.join(self.record.sig_name)}\n"
            info_text += f"\nESP32 Configuration:\n"
            info_text += f"- Gain: {self.gain}x\n"
            info_text += f"- ADC Resolution: {self.adc_resolution + 1} levels (0-{self.adc_resolution})\n"
            info_text += f"- VCC: {self.vcc}V\n"
            info_text += f"- Offset: {self.offset_voltage}V ({self.offset_adc} ADC)\n"
            self.info_sidebar.set_current_info(info_text)
            
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
        
        # Store time offset for x-axis display
        self.time_offset = self.trim_start
        
        # Trim signal and time
        self.signal_trimmed = self.signal[start_idx:end_idx]
        self.time_trimmed = self.time[start_idx:end_idx]  # Keep original time values
        
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
    
    def select_all_channels(self):
        """Select all channel checkboxes"""
        for checkbox in self.channel_checkboxes:
            checkbox.setChecked(True)
    
    def deselect_all_channels(self):
        """Deselect all channel checkboxes"""
        for checkbox in self.channel_checkboxes:
            checkbox.setChecked(False)
    
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
        visible_idx = 0
        last_visible_idx = -1
        
        # First, find the last visible channel
        for i in range(self.max_channels):
            if self.show_channel[i]:
                last_visible_idx = i
        
        for i, plot in enumerate(self.plot_widgets):
            if self.show_channel[i]:
                plot.setMinimumHeight(plot_height)
                plot.setMaximumHeight(plot_height if visible_count <= 3 else 16777215)
                
                # Show x-axis label only for the last visible plot
                if i == last_visible_idx:
                    plot.setLabel('bottom', 'Waktu (detik)')
                else:
                    plot.setLabel('bottom', '')
                
                plot.show()
                visible_idx += 1
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
    
    def check_data_warnings(self, mapped_signal):
        """Check for data warnings and empty channels"""
        warnings = []
        empty_channels = []
        
        # Check each channel
        for i in range(12):
            channel_data = mapped_signal[:, i]
            
            # Check for empty channels
            if np.all(channel_data == 0):
                empty_channels.append(self.standard_channels[i])
            else:
                # Check for clipping in original signal (assume ±6mV range)
                if np.any(channel_data > 6.0) or np.any(channel_data < -6.0):
                    warnings.append(f"Channel {self.standard_channels[i]}: Signal exceeds ±6mV range")
        
        if empty_channels:
            warnings.append(f"Empty channels detected: {', '.join(empty_channels)}")
        
        return warnings
    
    def convert_to_binary(self):
        """Convert ECG data to binary format with ESP32 compatibility"""
        if self.signal_trimmed is None:
            return
        
        try:
            # Clear previous warnings
            self.clipping_warnings = []
            
            # Map channels to standard 12-lead positions
            mapped_signal = self.map_channels_to_standard()
            
            # Check for warnings
            data_warnings = self.check_data_warnings(mapped_signal)
            
            # Convert each channel to ESP32 ADC values
            adc_signal = np.zeros_like(mapped_signal, dtype=np.uint16)
            
            for i in range(12):
                if np.any(mapped_signal[:, i] != 0):  # Only process non-empty channels
                    adc_signal[:, i] = self.convert_mv_to_adc(mapped_signal[:, i]).astype(np.uint16)
            
            # Create output filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"{self.record_combo.currentText()}_{timestamp}.bin"
            output_path = os.path.join(self.output_folder, output_filename)
            
            # Write binary file
            with open(output_path, 'wb') as f:
                for sample_idx in range(len(adc_signal)):
                    for channel_idx in range(12):
                        value = adc_signal[sample_idx, channel_idx]
                        f.write(struct.pack('<H', value))
            
            # Validate
            file_size = os.path.getsize(output_path)
            expected_size = len(adc_signal) * 12 * 2
            
            # Calculate statistics
            min_adc = np.min(adc_signal[adc_signal > 0]) if np.any(adc_signal > 0) else 0
            max_adc = np.max(adc_signal)
            min_voltage = self.convert_adc_to_voltage(min_adc)
            max_voltage = self.convert_adc_to_voltage(max_adc)
            
            # Prepare conversion info
            conversion_info = f"--- ESP32 Conversion Results ---\n"
            conversion_info += f"Output file: {output_filename}\n"
            conversion_info += f"Source record: {self.record_combo.currentText()}\n"
            conversion_info += f"Trimmed duration: {self.trim_start:.2f}s - {self.trim_end:.2f}s\n"
            conversion_info += f"Total duration: {self.trim_end - self.trim_start:.2f}s\n"
            conversion_info += f"Samples converted: {len(adc_signal):,}\n"
            conversion_info += f"File size: {file_size:,} bytes\n"
            conversion_info += f"Expected size: {expected_size:,} bytes\n"
            conversion_info += f"\nESP32 ADC Mapping:\n"
            conversion_info += f"- Gain applied: {self.gain}x\n"
            conversion_info += f"- ADC range: {min_adc} - {max_adc} (0-{self.adc_resolution})\n"
            conversion_info += f"- Voltage range: {min_voltage:.3f}V - {max_voltage:.3f}V (0-{self.vcc}V)\n"
            conversion_info += f"- Zero level: {self.offset_adc} ADC ({self.offset_voltage}V)\n"
            
            # Add warnings
            all_warnings = data_warnings + self.clipping_warnings
            if all_warnings:
                conversion_info += f"\n⚠️ WARNINGS:\n"
                for warning in all_warnings:
                    conversion_info += f"- {warning}\n"
            
            status = "Success" if file_size == expected_size else "Size mismatch"
            if all_warnings:
                status += " (with warnings)"
            
            conversion_info += f"\nStatus: {status}\n"
            
            # Update sidebar info
            current_info = self.info_sidebar.current_info.toPlainText()
            self.info_sidebar.set_current_info(current_info + "\n\n" + conversion_info)
            
            # Add to history
            history_item = ConversionHistoryItem(
                filename=output_filename,
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                duration=self.trim_end - self.trim_start,
                samples=len(adc_signal),
                file_size=file_size,
                status=status
            )
            self.info_sidebar.add_history_item(history_item)
            
            # Open sidebar to show results
            self.info_sidebar.open()
            
            # Show success message
            message = f"ESP32 Binary file created successfully!\n\n"
            message += f"File: {output_filename}\n"
            message += f"Size: {file_size:,} bytes\n"
            message += f"Duration: {self.trim_end - self.trim_start:.2f}s\n"
            message += f"ADC Range: {min_adc}-{max_adc}\n"
            message += f"Voltage Range: {min_voltage:.3f}V-{max_voltage:.3f}V\n"
            message += f"Location: {self.output_folder}\n"
            
            if all_warnings:
                message += f"\n⚠️ {len(all_warnings)} warning(s) detected. Check info panel for details."
                QMessageBox.warning(self, "Success with Warnings", message)
            else:
                QMessageBox.information(self, "Success", message)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Conversion failed: {str(e)}")
            
            # Add failed conversion to history
            history_item = ConversionHistoryItem(
                filename="Failed conversion",
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                duration=self.trim_end - self.trim_start if hasattr(self, 'trim_end') else 0,
                samples=0,
                file_size=0,
                status=f"Error: {str(e)}"
            )
            self.info_sidebar.add_history_item(history_item)
            self.info_sidebar.open()
    
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
                # Get raw data in mV
                raw_data = mapped_signal[self.current_index:end_idx, i]
                
                # Convert to display format based on Y-axis mode
                visible_data = self.get_display_data(raw_data)
                
                # Update plot - time axis shows actual time with offset
                self.plot_lines[i].setData(visible_time, visible_data)
                
                # Auto-scale based on current mode
                if len(visible_data) > 0 and np.any(visible_data != 0):
                    min_val = np.min(visible_data)
                    max_val = np.max(visible_data)
                    padding = (max_val - min_val) * 0.1 if max_val != min_val else 0.1
                    self.plot_widgets[i].setYRange(min_val - padding, max_val + padding)
                else:
                    # Set default range based on mode
                    if self.current_y_mode == 0:  # mV
                        self.plot_widgets[i].setYRange(-6, 6)
                    elif self.current_y_mode == 1:  # 12bit
                        self.plot_widgets[i].setYRange(0, self.adc_resolution)
                    else:  # Voltage
                        self.plot_widgets[i].setYRange(0, self.vcc)
        
        # Update x range with actual time values
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
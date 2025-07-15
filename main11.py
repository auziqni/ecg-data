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
from enum import Enum

class ECGMode(Enum):
    """ECG display modes"""
    TWELVE_LEAD = "12-lead"
    TEN_LEAD = "10-lead"

class ConversionHistoryItem:
    """Store conversion history data"""
    def __init__(self, filename, timestamp, duration, samples, file_size, status, mode):
        self.filename = filename
        self.timestamp = timestamp
        self.duration = duration
        self.samples = samples
        self.file_size = file_size
        self.status = status
        self.mode = mode  # Add mode info

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
            text = f"{item.timestamp} - {item.filename} [{item.mode}]\n"
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
        
        # ECG Mode
        self.ecg_mode = ECGMode.TWELVE_LEAD
        self.electrode_data = None  # Store 10-lead data
        self.conversion_errors = {}
        
        # Channel configurations
        self.CHANNELS_12LEAD = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 
                               'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        self.CHANNELS_10LEAD = ['RA', 'LA', 'LL', 'RL', 
                               'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        
        # Parameter plot
        self.window_size = 2000
        self.sample_rate = 360
        self.current_index = 0
        self.playing = False
        self.play_speed = 1.0
        
        # ESP32 ADC Configuration
        self.adc_resolution = 4095  # 12-bit ADC (0-4095)
        self.vcc = 3.3  # ESP32 VCC voltage
        self.gain = 1000  # Signal gain (default)
        self.offset_voltage = 1.65  # Offset voltage (VCC/2)
        self.offset_adc = 2048  # Offset in ADC counts (4095/2)
        
        # Y-axis display modes
        self.y_axis_modes = [
            "Asli (mV)",
            "Hasil (12bit)", 
            "Tegangan Hasil (V)"
        ]
        self.current_y_mode = 0
        
        # Grid lines control
        self.show_guide_lines = True
        
        # Maximum channels (always 12 for compatibility)
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
        self.setWindowTitle("ECG PhysioNet to Binary Converter - 12/10 Lead Mode")
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
        
        # Row 0: Record selection, Mode selection, Y-axis mode, convert button, and info panel toggle
        self.record_label = QLabel("Select Record:")
        self.control_layout.addWidget(self.record_label, 0, 0)
        
        self.record_combo = QComboBox()
        self.record_combo.currentTextChanged.connect(self.load_record)
        self.control_layout.addWidget(self.record_combo, 0, 1)
        
        # ECG Mode selector
        self.mode_label = QLabel("ECG Mode:")
        self.control_layout.addWidget(self.mode_label, 0, 2)
        
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["12-Lead Mode", "10-Lead Mode"])
        self.mode_combo.currentIndexChanged.connect(self.change_ecg_mode)
        self.control_layout.addWidget(self.mode_combo, 0, 3)
        
        # Y-axis mode selector
        self.y_mode_label = QLabel("Y-Axis:")
        self.control_layout.addWidget(self.y_mode_label, 0, 4)
        
        self.y_mode_combo = QComboBox()
        self.y_mode_combo.addItems(self.y_axis_modes)
        self.y_mode_combo.currentIndexChanged.connect(self.change_y_mode)
        self.control_layout.addWidget(self.y_mode_combo, 0, 5)
        
        self.convert_button = QPushButton("Convert to Binary")
        self.convert_button.clicked.connect(self.convert_to_binary)
        self.convert_button.setEnabled(False)
        self.control_layout.addWidget(self.convert_button, 0, 6)
        
        # Toggle info panel button
        self.info_toggle_btn = QPushButton("Info Panel")
        self.info_toggle_btn.clicked.connect(self.toggle_info_panel)
        self.control_layout.addWidget(self.info_toggle_btn, 0, 7)
        
        # Guide lines toggle
        self.guide_toggle_btn = QPushButton("Guide: ON")
        self.guide_toggle_btn.clicked.connect(self.toggle_guide_lines)
        self.guide_toggle_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; }")
        self.control_layout.addWidget(self.guide_toggle_btn, 0, 8)
        
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
        
        # Conversion status label
        self.conversion_status_label = QLabel("")
        self.conversion_status_label.setStyleSheet("color: blue; font-weight: bold;")
        self.control_layout.addWidget(self.conversion_status_label, 1, 6, 1, 3)
        
        # Row 2: Window size, gain control, and ESP32 info
        self.window_label = QLabel("Window Size:")
        self.control_layout.addWidget(self.window_label, 2, 0)
        
        self.window_spinbox = QSpinBox()
        self.window_spinbox.setMinimum(100)
        self.window_spinbox.setMaximum(10000)
        self.window_spinbox.setSingleStep(100)
        self.window_spinbox.setValue(2000)
        self.window_spinbox.valueChanged.connect(self.change_window_size)
        self.control_layout.addWidget(self.window_spinbox, 2, 1)
        
        # Gain control
        self.gain_label = QLabel("Gain:")
        self.control_layout.addWidget(self.gain_label, 2, 2)
        
        self.gain_spinbox = QSpinBox()
        self.gain_spinbox.setMinimum(1)
        self.gain_spinbox.setMaximum(10000)
        self.gain_spinbox.setSingleStep(10)
        self.gain_spinbox.setValue(1000)
        self.gain_spinbox.valueChanged.connect(self.change_gain)
        self.control_layout.addWidget(self.gain_spinbox, 2, 3)
        
        self.sample_rate_label = QLabel("Sample Rate: -")
        self.control_layout.addWidget(self.sample_rate_label, 2, 4)
        
        self.duration_label = QLabel("Duration: -")
        self.control_layout.addWidget(self.duration_label, 2, 5)
        
        # Row 3: ESP32 info and gain warning
        self.esp32_info_label = QLabel(f"ESP32: Gain={self.gain}x, ADC=12bit, VCC=3.3V")
        self.esp32_info_label.setStyleSheet("color: blue; font-weight: bold;")
        self.control_layout.addWidget(self.esp32_info_label, 3, 0, 1, 4)
        
        # Gain warning label
        self.gain_warning_label = QLabel("")
        self.gain_warning_label.setStyleSheet("color: #FF8C00; font-weight: bold;")  # Orange/yellow color
        self.gain_warning_label.setVisible(False)
        self.control_layout.addWidget(self.gain_warning_label, 3, 4, 1, 3)
        
        # Row 4: Trim controls
        self.trim_label = QLabel("Trim Signal:")
        self.control_layout.addWidget(self.trim_label, 4, 0)
        
        self.start_label = QLabel("Start (s):")
        self.control_layout.addWidget(self.start_label, 4, 1)
        
        self.start_spinbox = QDoubleSpinBox()
        self.start_spinbox.setMinimum(0.0)
        self.start_spinbox.setMaximum(9999.0)
        self.start_spinbox.setDecimals(2)
        self.start_spinbox.setSingleStep(0.1)
        self.start_spinbox.valueChanged.connect(self.update_trim)
        self.control_layout.addWidget(self.start_spinbox, 4, 2)
        
        self.end_label = QLabel("End (s):")
        self.control_layout.addWidget(self.end_label, 4, 3)
        
        self.end_spinbox = QDoubleSpinBox()
        self.end_spinbox.setMinimum(0.0)
        self.end_spinbox.setMaximum(9999.0)
        self.end_spinbox.setDecimals(2)
        self.end_spinbox.setSingleStep(0.1)
        self.end_spinbox.valueChanged.connect(self.update_trim)
        self.control_layout.addWidget(self.end_spinbox, 4, 4)
        
        self.trim_info_label = QLabel("Trimmed: - samples")
        self.control_layout.addWidget(self.trim_info_label, 4, 5)
    
    def change_ecg_mode(self, index):
        """Change between 12-lead and 10-lead mode"""
        if index == 0:
            self.ecg_mode = ECGMode.TWELVE_LEAD
            self.conversion_status_label.setText("Mode: 12-Lead Standard")
            self.conversion_status_label.setStyleSheet("color: blue; font-weight: bold;")
        else:
            self.ecg_mode = ECGMode.TEN_LEAD
            self.conversion_status_label.setText("Mode: 10-Lead Raw Electrodes")
            self.conversion_status_label.setStyleSheet("color: green; font-weight: bold;")
            
            # Convert data if available
            if self.signal_trimmed is not None:
                self.convert_to_10lead()
        
        # Update channel display
        self.update_channel_display()
        self.update_plots()
        
        # Update info panel
        self.update_info_panel()
    
    def convert_to_10lead(self):
        """Convert 12-lead data to 10-lead raw electrodes"""
        if self.signal_trimmed is None:
            return
        
        # Map channels to standard 12-lead positions
        mapped_signal = self.map_channels_to_standard()
        
        # Extract leads for conversion
        lead_I = mapped_signal[:, 0]
        lead_II = mapped_signal[:, 1]
        lead_III = mapped_signal[:, 2]
        
        # Calculate electrodes with RL = 0 and WCT = 0
        RA = -(lead_I + lead_II) / 3
        LA = (2 * lead_I - lead_II) / 3
        LL = (2 * lead_II - lead_I) / 3
        RL = np.zeros_like(RA)
        
        # Validate conversion
        lead_III_calc = LL - LA
        error = np.mean(np.abs(lead_III_calc - lead_III))
        
        if error > 0.05:  # 50 μV threshold
            self.conversion_errors['Lead III'] = error
            warning_msg = f"⚠️ Conversion validation warning: Lead III error = {error:.3f} mV"
            self.statusBar().showMessage(warning_msg)
        else:
            self.conversion_errors = {}
            self.statusBar().showMessage(f"10-lead conversion successful. Error: {error:.4f} mV")
        
        # Create 10-lead electrode data
        self.electrode_data = np.zeros((len(mapped_signal), 10))
        self.electrode_data[:, 0] = RA
        self.electrode_data[:, 1] = LA
        self.electrode_data[:, 2] = LL
        self.electrode_data[:, 3] = RL
        self.electrode_data[:, 4:10] = mapped_signal[:, 6:12]  # V1-V6
        
        return error
    
    def update_channel_display(self):
        """Update channel labels and visibility based on mode"""
        if self.ecg_mode == ECGMode.TWELVE_LEAD:
            # 12-lead mode: show all 12 channels
            channels = self.CHANNELS_12LEAD
            max_visible = 12
        else:
            # 10-lead mode: show only 10 channels
            channels = self.CHANNELS_10LEAD
            max_visible = 10
        
        # Update checkboxes
        for i in range(self.max_channels):
            if i < len(channels):
                self.channel_checkboxes[i].setText(channels[i])
                self.channel_checkboxes[i].setVisible(True)
                self.channel_checkboxes[i].setEnabled(True)
                
                # Update plot title
                self.plot_widgets[i].setTitle(channels[i])
                self.plot_widgets[i].setLabel('left', self.get_y_label(i))
            else:
                self.channel_checkboxes[i].setVisible(False)
                self.channel_checkboxes[i].setEnabled(False)
                self.plot_widgets[i].setVisible(False)
                self.show_channel[i] = False
        
        # Update plot layout
        self.update_plot_layout()
    
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
            # Use 12-lead labels initially
            label = self.CHANNELS_12LEAD[i] if i < len(self.CHANNELS_12LEAD) else f"Ch{i+1}"
            checkbox = QCheckBox(label)
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
        self.guide_lines_x = []  # X-axis guide lines (time = 0)
        self.guide_lines_y_zero = []  # Y-axis guide lines (value = 0)
        self.guide_lines_y_offset = []  # Y-axis guide lines (offset values)
        
        for i in range(self.max_channels):
            # Create plot widget
            plot_widget = pg.PlotWidget()
            plot_widget.setBackground('w')
            plot_widget.showGrid(x=True, y=True, alpha=0.3)
            plot_widget.setLabel('left', self.get_y_label(i))
            
            # Always show x-axis values for all plots, but only label the last one
            plot_widget.getAxis('bottom').setStyle(showValues=True)
            
            # Set initial title
            if i < len(self.CHANNELS_12LEAD):
                plot_widget.setTitle(self.CHANNELS_12LEAD[i])
            
            # Plot line for data
            pen = pg.mkPen(color=self.channel_colors[i], width=2)
            plot_line = plot_widget.plot(pen=pen)
            
            # Guide lines
            # X-axis guide line (time = 0, solid black)
            pen_x_guide = pg.mkPen(color='black', width=1, style=1)  # solid
            guide_line_x = plot_widget.addLine(x=0, pen=pen_x_guide)
            
            # Y-axis guide line (value = 0, solid black)
            pen_y_zero_guide = pg.mkPen(color='black', width=1, style=1)  # solid
            guide_line_y_zero = plot_widget.addLine(y=0, pen=pen_y_zero_guide)
            
            # Y-axis offset guide line (dashed black)
            pen_y_offset_guide = pg.mkPen(color='black', width=1, style=2)  # dashed
            guide_line_y_offset = plot_widget.addLine(y=2048, pen=pen_y_offset_guide)
            
            self.plots_layout.addWidget(plot_widget)
            self.plot_widgets.append(plot_widget)
            self.plot_lines.append(plot_line)
            self.guide_lines_x.append(guide_line_x)
            self.guide_lines_y_zero.append(guide_line_y_zero)
            self.guide_lines_y_offset.append(guide_line_y_offset)
            
            plot_widget.setVisible(False)
        
        # Link X axes
        for i in range(1, self.max_channels):
            self.plot_widgets[i].setXLink(self.plot_widgets[0])
    
    def get_y_label(self, channel_idx):
        """Get Y-axis label based on current mode"""
        if self.ecg_mode == ECGMode.TWELVE_LEAD:
            channels = self.CHANNELS_12LEAD
        else:
            channels = self.CHANNELS_10LEAD
            
        if channel_idx < len(channels):
            channel = channels[channel_idx]
        else:
            channel = f"Ch{channel_idx+1}"
            
        mode = self.y_axis_modes[self.current_y_mode]
        return f"{channel}\n({mode})"
    
    def toggle_guide_lines(self):
        """Toggle guide lines visibility"""
        self.show_guide_lines = not self.show_guide_lines
        
        # Update button appearance
        if self.show_guide_lines:
            self.guide_toggle_btn.setText("Guide: ON")
            self.guide_toggle_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; }")
        else:
            self.guide_toggle_btn.setText("Guide: OFF")
            self.guide_toggle_btn.setStyleSheet("QPushButton { background-color: #f44336; color: white; }")
        
        # Update guide lines visibility
        self.update_guide_lines()
    
    def update_guide_lines(self):
        """Update guide lines based on current mode and toggle state"""
        for i in range(self.max_channels):
            if not self.show_channel[i]:
                continue
                
            # X-axis guide line (time = 0) - always shown if guide lines enabled
            self.guide_lines_x[i].setVisible(self.show_guide_lines)
            
            # Y-axis guide line (value = 0) - always shown if guide lines enabled
            self.guide_lines_y_zero[i].setVisible(self.show_guide_lines)
            
            # Y-axis offset guide line - depends on mode
            if self.show_guide_lines:
                if self.current_y_mode == 0:  # Asli (mV) - no offset line
                    self.guide_lines_y_offset[i].setVisible(False)
                elif self.current_y_mode == 1:  # Hasil (12bit) - line at 2048
                    self.guide_lines_y_offset[i].setValue(2048)
                    self.guide_lines_y_offset[i].setVisible(True)
                elif self.current_y_mode == 2:  # Tegangan Hasil (V) - line at 1.65
                    self.guide_lines_y_offset[i].setValue(1.65)
                    self.guide_lines_y_offset[i].setVisible(True)
            else:
                self.guide_lines_y_offset[i].setVisible(False)
    
    def check_gain_warnings(self):
        """Check if current gain will cause signal clipping with actual data in trim range only"""
        if self.signal_trimmed is None:
            self.gain_warning_label.setVisible(False)
            self.convert_button.setEnabled(False)
            return
        
        # Get appropriate data based on mode
        if self.ecg_mode == ECGMode.TEN_LEAD and self.electrode_data is not None:
            # For 10-lead mode, check electrode data
            signal_to_check = self.electrode_data
        else:
            # For 12-lead mode, check mapped signal
            signal_to_check = self.map_channels_to_standard()
        
        # Check if there's any actual signal data (non-zero)
        has_data = np.any(signal_to_check != 0)
        if not has_data:
            self.gain_warning_label.setVisible(False)
            self.convert_button.setEnabled(True)
            return
        
        # Find min/max of actual signal data
        non_zero_data = signal_to_check[signal_to_check != 0]
        if len(non_zero_data) == 0:
            self.gain_warning_label.setVisible(False)
            self.convert_button.setEnabled(True)
            return
            
        min_signal = np.min(non_zero_data)
        max_signal = np.max(non_zero_data)
        
        # Apply gain and calculate voltage range
        min_voltage_after_gain = (min_signal * self.gain / 1000.0) + self.offset_voltage
        max_voltage_after_gain = (max_signal * self.gain / 1000.0) + self.offset_voltage
        
        # Check if signal will exceed ESP32 limits
        voltage_overflow = max_voltage_after_gain > self.vcc or min_voltage_after_gain < 0
        
        if voltage_overflow:
            # Show warning with current range vs expected range
            warning_text = f"⚠️ WARNING! Range: {min_voltage_after_gain:.2f}V to {max_voltage_after_gain:.2f}V "
            warning_text += f"(Should be: 0V to {self.vcc}V) [Mode: {self.ecg_mode.value}]"
            
            self.gain_warning_label.setText(warning_text)
            self.gain_warning_label.setVisible(True)
            
            # Disable conversion when signal is out of range
            self.convert_button.setEnabled(False)
            self.convert_button.setText("Convert Disabled (Out of Range)")
        else:
            self.gain_warning_label.setVisible(False)
            
            # Enable conversion when signal is in acceptable range
            self.convert_button.setEnabled(True)
            self.convert_button.setText("Convert to Binary")
    
    def change_gain(self, value):
        """Change gain value and update display"""
        self.gain = value
        
        # Update ESP32 info label
        self.esp32_info_label.setText(f"ESP32: Gain={self.gain}x, ADC=12bit, VCC=3.3V")
        
        # Check for gain warnings with actual data
        self.check_gain_warnings()
        
        # Update plots with new gain
        self.update_plots()
        
        # Clear previous warnings since gain changed
        self.clipping_warnings = []
    
    def change_y_mode(self, index):
        """Change Y-axis display mode"""
        self.current_y_mode = index
        
        # Update all plot labels
        for i in range(self.max_channels):
            self.plot_widgets[i].setLabel('left', self.get_y_label(i))
        
        # Update guide lines for new mode
        self.update_guide_lines()
        
        # Update plots with new scaling
        self.update_plots()
    
    def convert_mv_to_adc(self, signal_mv, apply_clipping=True):
        """Convert mV signal to ESP32 ADC values"""
        # Apply gain (gain value)
        signal_gained = signal_mv * self.gain / 1000.0  # Convert mV to V and apply gain
        
        # Add offset (1.65V)
        signal_offset = signal_gained + self.offset_voltage
        
        # Convert to ADC counts
        adc_values = signal_offset * self.adc_resolution / self.vcc
        
        # Apply clipping only if requested (for conversion, not for display)
        if apply_clipping:
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
            # For display, don't apply clipping - show actual calculated values
            return self.convert_mv_to_adc(signal_mv, apply_clipping=False)
        else:  # Tegangan Hasil (V)
            adc_values = self.convert_mv_to_adc(signal_mv, apply_clipping=False)
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
    
    def update_info_panel(self):
        """Update info panel with current mode information"""
        if self.record is None:
            return
            
        info_text = f"Record: {self.record_combo.currentText()}\n"
        info_text += f"Mode: {self.ecg_mode.value}\n"
        info_text += f"Channels available: {self.signal.shape[1] if len(self.signal.shape) > 1 else 1}\n"
        info_text += f"Sample rate: {self.sample_rate} Hz\n"
        info_text += f"Total samples: {len(self.signal):,}\n"
        
        if self.ecg_mode == ECGMode.TWELVE_LEAD:
            info_text += f"\nChannel names (12-lead):\n"
            info_text += f"{', '.join(self.CHANNELS_12LEAD)}\n"
        else:
            info_text += f"\nChannel names (10-lead electrodes):\n"
            info_text += f"{', '.join(self.CHANNELS_10LEAD)}\n"
            
            if self.conversion_errors:
                info_text += f"\n⚠️ Conversion Errors:\n"
                for lead, error in self.conversion_errors.items():
                    info_text += f"- {lead}: {error:.3f} mV\n"
        
        info_text += f"\nESP32 Configuration:\n"
        info_text += f"- Gain: {self.gain}x\n"
        info_text += f"- ADC Resolution: {self.adc_resolution + 1} levels\n"
        info_text += f"- VCC: {self.vcc}V\n"
        info_text += f"- Offset: {self.offset_voltage}V ({self.offset_adc} ADC)\n"
        
        self.info_sidebar.set_current_info(info_text)
    
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
            self.conversion_errors = {}
            
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
            
            # If in 10-lead mode, convert immediately
            if self.ecg_mode == ECGMode.TEN_LEAD:
                self.convert_to_10lead()
            
            # Update UI
            self.current_index = 0
            self.update_record_info()
            self.update_plots()
            self.convert_button.setEnabled(True)
            
            # Update info panel
            self.update_info_panel()
            
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
        
        # If in 10-lead mode, reconvert
        if self.ecg_mode == ECGMode.TEN_LEAD:
            self.convert_to_10lead()
        
        # Reset playback and update plots
        self.current_index = 0
        self.check_gain_warnings()  # Recheck warnings with trimmed data
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
            if checkbox.isVisible():
                checkbox.setChecked(True)
    
    def deselect_all_channels(self):
        """Deselect all channel checkboxes"""
        for checkbox in self.channel_checkboxes:
            if checkbox.isVisible():
                checkbox.setChecked(False)
    
    def toggle_channel(self, channel, state):
        """Toggle channel visibility"""
        self.show_channel[channel] = state > 0
        self.update_plot_layout()
        self.update_plots()
    
    def update_plot_layout(self):
        """Update plot layout based on visible channels"""
        # Count visible channels based on mode
        if self.ecg_mode == ECGMode.TEN_LEAD:
            max_channels = 10
        else:
            max_channels = 12
        
        visible_count = sum(self.show_channel[:max_channels])
        
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
        for i in range(max_channels):
            if self.show_channel[i]:
                last_visible_idx = i
        
        for i, plot in enumerate(self.plot_widgets):
            if i < max_channels and self.show_channel[i]:
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
        
        # Update guide lines visibility
        self.update_guide_lines()
    
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
            
            for j, std_ch in enumerate(self.CHANNELS_12LEAD):
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
                empty_channels.append(self.CHANNELS_12LEAD[i])
            else:
                # Check for clipping in original signal (assume ±9mV range)
                if np.any(channel_data > 9.0) or np.any(channel_data < -9.0):
                    warnings.append(f"Channel {self.CHANNELS_12LEAD[i]}: Signal exceeds ±9mV range")
        
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
            
            # Prepare data based on mode
            if self.ecg_mode == ECGMode.TWELVE_LEAD:
                # 12-lead mode: use mapped signal directly
                mapped_signal = self.map_channels_to_standard()
                adc_signal = np.zeros((len(mapped_signal), 12), dtype=np.uint16)
                
                # Convert each channel to ESP32 ADC values
                for i in range(12):
                    if np.any(mapped_signal[:, i] != 0):  # Only process non-empty channels
                        adc_signal[:, i] = self.convert_mv_to_adc(mapped_signal[:, i], apply_clipping=True).astype(np.uint16)
                
                # Check for warnings
                data_warnings = self.check_data_warnings(mapped_signal)
                
            else:  # ECGMode.TEN_LEAD
                # 10-lead mode: arrange as RA,LA,LL,RL,0,0,V1-V6
                if self.electrode_data is None:
                    QMessageBox.warning(self, "Error", "No 10-lead data available. Please reload the record.")
                    return
                
                adc_signal = np.zeros((len(self.electrode_data), 12), dtype=np.uint16)
                
                # Convert limb electrodes (RA, LA, LL, RL)
                for i in range(4):
                    adc_signal[:, i] = self.convert_mv_to_adc(self.electrode_data[:, i], apply_clipping=True).astype(np.uint16)
                
                # Channels 4 and 5 are padding (zeros)
                adc_signal[:, 4:6] = 0
                
                # Convert chest electrodes (V1-V6)
                for i in range(6):
                    adc_signal[:, i+6] = self.convert_mv_to_adc(self.electrode_data[:, i+4], apply_clipping=True).astype(np.uint16)
                
                data_warnings = []
                if self.conversion_errors:
                    for lead, error in self.conversion_errors.items():
                        data_warnings.append(f"Conversion error - {lead}: {error:.3f} mV")
            
            # Create output filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            mode_suffix = "10lead" if self.ecg_mode == ECGMode.TEN_LEAD else "12lead"
            output_filename = f"{self.record_combo.currentText()}_{mode_suffix}_{timestamp}.bin"
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
            conversion_info += f"Mode: {self.ecg_mode.value}\n"
            conversion_info += f"Source record: {self.record_combo.currentText()}\n"
            conversion_info += f"Trimmed duration: {self.trim_start:.2f}s - {self.trim_end:.2f}s\n"
            conversion_info += f"Total duration: {self.trim_end - self.trim_start:.2f}s\n"
            conversion_info += f"Samples converted: {len(adc_signal):,}\n"
            conversion_info += f"File size: {file_size:,} bytes\n"
            conversion_info += f"Expected size: {expected_size:,} bytes\n"
            
            if self.ecg_mode == ECGMode.TEN_LEAD:
                conversion_info += f"\n10-Lead Channel Mapping:\n"
                conversion_info += f"Ch 1-4: RA, LA, LL, RL (electrodes)\n"
                conversion_info += f"Ch 5-6: Padding (zeros)\n"
                conversion_info += f"Ch 7-12: V1-V6\n"
            
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
                status=status,
                mode=self.ecg_mode.value
            )
            self.info_sidebar.add_history_item(history_item)
            
            # Open sidebar to show results
            self.info_sidebar.open()
            
            # Show success message
            message = f"ESP32 Binary file created successfully!\n\n"
            message += f"Mode: {self.ecg_mode.value}\n"
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
                status=f"Error: {str(e)}",
                mode=self.ecg_mode.value
            )
            self.info_sidebar.add_history_item(history_item)
            self.info_sidebar.open()
    
    def update_plots(self):
        """Update all plots with current window"""
        if self.signal_trimmed is None:
            return
        
        # Get appropriate data based on mode
        if self.ecg_mode == ECGMode.TWELVE_LEAD:
            # 12-lead mode: use mapped signal
            display_signal = self.map_channels_to_standard()
            num_channels = 12
        else:  # TEN_LEAD
            # 10-lead mode: use electrode data if available
            if self.electrode_data is not None:
                display_signal = self.electrode_data
                num_channels = 10
            else:
                # Fallback to empty data
                display_signal = np.zeros((len(self.signal_trimmed), 10))
                num_channels = 10
        
        end_idx = min(self.current_index + self.window_size, len(self.signal_trimmed))
        visible_time = self.time_trimmed[self.current_index:end_idx]
        
        # Update each channel
        for i in range(self.max_channels):
            if i < num_channels and self.show_channel[i]:
                # Get raw data in mV
                raw_data = display_signal[self.current_index:end_idx, i]
                
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
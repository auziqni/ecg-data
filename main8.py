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

# Add Y-axis unit options and conversion functions
UNITS = {
    'asli(mv)': {'name': 'mV', 'convert': lambda x, atten: x},
    'hasil(12bit)': {'name': 'ADC', 'convert': lambda x, atten: (x/atten + 1.65) * (4095/3.3)},
    'tegangan hasil(v)': {'name': 'V', 'convert': lambda x, atten: x/atten + 1.65}
}

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
        
        # Y-axis unit selection
        self.current_unit = 'asli(mv)'
        self.attenuation = 1000.0
        
        # Trim parameters
        self.trim_start = 0.0
        self.trim_end = 0.0
        self.time_offset = 0.0
        
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
        
        # Row 0: Record selection, Y-axis unit, convert button, and info panel toggle
        self.record_label = QLabel("Select Record:")
        self.control_layout.addWidget(self.record_label, 0, 0)
        
        self.record_combo = QComboBox()
        self.record_combo.currentTextChanged.connect(self.load_record)
        self.control_layout.addWidget(self.record_combo, 0, 1)
        
        # Add Y-axis unit selection and attenuation
        self.unit_label = QLabel("Y-axis Unit:")
        self.control_layout.addWidget(self.unit_label, 0, 2)
        
        self.unit_combo = QComboBox()
        self.unit_combo.addItems(UNITS.keys())
        self.unit_combo.currentTextChanged.connect(self.change_unit)
        self.control_layout.addWidget(self.unit_combo, 0, 3)

        self.atten_label = QLabel("Attenuation:")
        self.control_layout.addWidget(self.atten_label, 0, 4)
        
        self.atten_spinbox = QDoubleSpinBox()
        self.atten_spinbox.setMinimum(1.0)
        self.atten_spinbox.setMaximum(100000.0)
        self.atten_spinbox.setValue(1000.0)
        self.atten_spinbox.setSingleStep(100.0)
        self.atten_spinbox.valueChanged.connect(self.change_attenuation)
        self.control_layout.addWidget(self.atten_spinbox, 0, 5)
        
        # Convert and info panel buttons in a horizontal layout
        button_layout = QHBoxLayout()
        
        self.convert_button = QPushButton("Convert to Binary")
        self.convert_button.clicked.connect(self.convert_to_binary)
        self.convert_button.setEnabled(False)
        button_layout.addWidget(self.convert_button)
        
        self.info_toggle_btn = QPushButton("Show/Hide Info Panel")
        self.info_toggle_btn.clicked.connect(self.toggle_info_panel)
        button_layout.addWidget(self.info_toggle_btn)
        
        # Add button layout to grid
        button_widget = QWidget()
        button_widget.setLayout(button_layout)
        self.control_layout.addWidget(button_widget, 0, 6, 1, 2)
        
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
        self.control_layout.addWidget(self.speed_slider, 1, 3, 1, 4)
        
        self.speed_value = QLabel("1.0x")
        self.control_layout.addWidget(self.speed_value, 1, 7)
        
        # Row 2: Window size and info
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
            plot_widget.setLabel('left', self.standard_channels[i])
            
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
        
        # Update total samples
        self.total_samples_label.setText(f"Total: {len(self.signal):,} samples")

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

    def change_unit(self, unit):
        """Change Y-axis unit display"""
        self.current_unit = unit
        self.update_plots()
        
    def change_attenuation(self, value):
        """Change signal attenuation value"""
        self.attenuation = value
        self.update_plots()

    def update_plots(self):
        """Update all plots with current window"""
        if self.signal_trimmed is None:
            return
        
        # Map channels for display
        mapped_signal = self.map_channels_to_standard()
        
        end_idx = min(self.current_index + self.window_size, len(self.signal_trimmed))
        visible_time = self.time_trimmed[self.current_index:end_idx]
        
        # Get conversion function for current unit
        convert_func = UNITS[self.current_unit]['convert']
        unit_name = UNITS[self.current_unit]['name']
        
        # Update each channel
        for i in range(self.max_channels):
            if self.show_channel[i]:
                visible_data = mapped_signal[self.current_index:end_idx, i]
                
                # Convert data to selected unit with current attenuation
                visible_data = convert_func(visible_data, self.attenuation)
                
                # Update plot
                self.plot_lines[i].setData(visible_time, visible_data)
                
                # Update y-axis label with unit
                self.plot_widgets[i].setLabel('left', f'{self.standard_channels[i]} ({unit_name})')
                
                # Auto-scale
                if len(visible_data) > 0 and np.any(visible_data != 0):
                    min_val = np.min(visible_data)
                    max_val = np.max(visible_data)
                    padding = (max_val - min_val) * 0.1
                    self.plot_widgets[i].setYRange(min_val - padding, max_val + padding)
        
        # Update x range with actual time values
        if len(visible_time) > 0:
            self.plot_widgets[0].setXRange(visible_time[0], visible_time[-1])

    def convert_to_binary(self):
        """Convert ECG data to binary format"""
        if self.signal_trimmed is None:
            return
        
        try:
            # Map channels to standard 12-lead positions
            mapped_signal = self.map_channels_to_standard()
            
            # Process signal for ESP32 ADC:
            # 1. Apply attenuation
            attenuated_signal = mapped_signal / self.attenuation
            
            # 2. Shift to center at 1.65V and scale to 0-3.3V range
            voltage_signal = attenuated_signal + 1.65
            
            # 3. Convert to 12-bit ADC values (0-4095)
            adc_signal = (voltage_signal * (4095/3.3)).round().clip(0, 4095).astype(np.uint16)
            
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
            
            # Prepare conversion info
            conversion_info = f"--- Conversion Results ---\n"
            conversion_info += f"Output file: {output_filename}\n"
            conversion_info += f"Source record: {self.record_combo.currentText()}\n"
            conversion_info += f"Trimmed duration: {self.trim_start:.2f}s - {self.trim_end:.2f}s\n"
            conversion_info += f"Total duration: {self.trim_end - self.trim_start:.2f}s\n"
            conversion_info += f"Samples converted: {len(adc_signal):,}\n"
            conversion_info += f"File size: {file_size:,} bytes\n"
            conversion_info += f"Expected size: {expected_size:,} bytes\n"
            conversion_info += f"ADC Range: 0-4095 (centered at 2048)\n"
            conversion_info += f"Voltage Range: 0-3.3V (centered at 1.65V)\n"
            conversion_info += f"Attenuation: {self.attenuation:.1f}x\n"
            conversion_info += f"Status: {'Success' if file_size == expected_size else 'Size mismatch!'}\n"
            
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
                status="Success" if file_size == expected_size else "Size mismatch"
            )
            self.info_sidebar.add_history_item(history_item)
            
            # Open sidebar to show results
            self.info_sidebar.open()
            
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
            
            # Add failed conversion to history
            history_item = ConversionHistoryItem(
                filename="Failed conversion",
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                duration=self.trim_end - self.trim_start,
                samples=0,
                file_size=0,
                status=f"Error: {str(e)}"
            )
            self.info_sidebar.add_history_item(history_item)
            self.info_sidebar.open()

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

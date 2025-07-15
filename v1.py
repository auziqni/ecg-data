import sys
import numpy as np
import pandas as pd
import pyqtgraph as pg
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QWidget, 
                            QPushButton, QHBoxLayout, QLabel, QComboBox, 
                            QCheckBox, QSlider, QSpinBox, QGridLayout, QGroupBox,
                            QDoubleSpinBox, QFileDialog, QMessageBox, QRadioButton,
                            QButtonGroup, QSplitter)
from PyQt5.QtCore import QTimer, Qt, pyqtSignal
from PyQt5.QtGui import QFont
import os
import struct
from scipy import signal as scipy_signal
from scipy.interpolate import interp1d

class ECGSignalValidator(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # ESP32 Configuration (same as converter)
        self.adc_resolution = 4095
        self.vcc = 3.3
        self.offset_voltage = 1.65
        self.offset_adc = 2048
        self.gain = 1000  # Default gain
        
        # Sample rates
        self.reference_sample_rate = 360  # From PhysioNet
        self.measured_sample_rate = 1000  # Default oscilloscope rate
        
        # Standard channel names
        self.channel_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 
                             'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        self.current_channel = 0
        
        # Data storage
        self.reference_data = None  # 12-channel reference from binary
        self.measured_data = {}     # Dictionary of measured signals by channel
        self.time_reference = None
        self.time_measured = {}
        
        # Display mode
        self.display_mode = "side_by_side"  # or "overlay"
        
        # Plot parameters
        self.window_size = 2000
        self.current_index = 0
        self.time_offset = 0.0
        
        # Signal folder
        self.signal_folder = "signal"
        
        # UI Setup
        self.setWindowTitle("ECG Signal Validation Tool")
        self.setGeometry(100, 50, 1400, 900)
        
        # Main widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        
        # Create UI components
        self.create_control_panel()
        self.create_plot_area()
        
        # Load data
        self.load_all_data()
        
        # Update initial display
        self.update_plots()
    
    def create_control_panel(self):
        # Control panel
        self.control_group = QGroupBox("Controls")
        self.main_layout.addWidget(self.control_group)
        self.control_layout = QGridLayout(self.control_group)
        
        # Row 0: File loading and channel selection
        self.load_button = QPushButton("Reload Data")
        self.load_button.clicked.connect(self.load_all_data)
        self.control_layout.addWidget(self.load_button, 0, 0)
        
        self.channel_label = QLabel("Channel:")
        self.control_layout.addWidget(self.channel_label, 0, 1)
        
        self.channel_combo = QComboBox()
        self.channel_combo.addItems(self.channel_names)
        self.channel_combo.currentIndexChanged.connect(self.change_channel)
        self.control_layout.addWidget(self.channel_combo, 0, 2)
        
        # Display mode selection
        self.display_label = QLabel("Display Mode:")
        self.control_layout.addWidget(self.display_label, 0, 3)
        
        self.display_mode_group = QButtonGroup()
        self.side_by_side_radio = QRadioButton("Side by Side")
        self.side_by_side_radio.setChecked(True)
        self.overlay_radio = QRadioButton("Overlay")
        
        self.display_mode_group.addButton(self.side_by_side_radio)
        self.display_mode_group.addButton(self.overlay_radio)
        
        self.side_by_side_radio.toggled.connect(self.change_display_mode)
        
        self.control_layout.addWidget(self.side_by_side_radio, 0, 4)
        self.control_layout.addWidget(self.overlay_radio, 0, 5)
        
        # Row 1: ESP32 parameters
        self.esp32_label = QLabel("ESP32 Parameters:")
        self.esp32_label.setStyleSheet("font-weight: bold;")
        self.control_layout.addWidget(self.esp32_label, 1, 0)
        
        self.gain_label = QLabel("Gain:")
        self.control_layout.addWidget(self.gain_label, 1, 1)
        
        self.gain_spinbox = QSpinBox()
        self.gain_spinbox.setMinimum(1)
        self.gain_spinbox.setMaximum(10000)
        self.gain_spinbox.setValue(1000)
        self.gain_spinbox.valueChanged.connect(self.update_gain)
        self.control_layout.addWidget(self.gain_spinbox, 1, 2)
        
        self.offset_label = QLabel("Offset (V):")
        self.control_layout.addWidget(self.offset_label, 1, 3)
        
        self.offset_spinbox = QDoubleSpinBox()
        self.offset_spinbox.setMinimum(0.0)
        self.offset_spinbox.setMaximum(3.3)
        self.offset_spinbox.setDecimals(3)
        self.offset_spinbox.setValue(1.65)
        self.offset_spinbox.valueChanged.connect(self.update_offset)
        self.control_layout.addWidget(self.offset_spinbox, 1, 4)
        
        # Row 2: Sample rates
        self.sample_rate_label = QLabel("Sample Rates:")
        self.sample_rate_label.setStyleSheet("font-weight: bold;")
        self.control_layout.addWidget(self.sample_rate_label, 2, 0)
        
        self.ref_rate_label = QLabel("Reference:")
        self.control_layout.addWidget(self.ref_rate_label, 2, 1)
        
        self.ref_rate_display = QLabel(f"{self.reference_sample_rate} Hz")
        self.control_layout.addWidget(self.ref_rate_display, 2, 2)
        
        self.measured_rate_label = QLabel("Measured:")
        self.control_layout.addWidget(self.measured_rate_label, 2, 3)
        
        self.measured_rate_spinbox = QSpinBox()
        self.measured_rate_spinbox.setMinimum(1)
        self.measured_rate_spinbox.setMaximum(1000000)
        self.measured_rate_spinbox.setValue(1000)
        self.measured_rate_spinbox.setSuffix(" Hz")
        self.measured_rate_spinbox.valueChanged.connect(self.update_measured_sample_rate)
        self.control_layout.addWidget(self.measured_rate_spinbox, 2, 4)
        
        # Row 3: Time offset and window controls
        self.offset_time_label = QLabel("Time Offset (s):")
        self.control_layout.addWidget(self.offset_time_label, 3, 0)
        
        self.time_offset_spinbox = QDoubleSpinBox()
        self.time_offset_spinbox.setMinimum(-10.0)
        self.time_offset_spinbox.setMaximum(10.0)
        self.time_offset_spinbox.setDecimals(3)
        self.time_offset_spinbox.setSingleStep(0.001)
        self.time_offset_spinbox.setValue(0.0)
        self.time_offset_spinbox.valueChanged.connect(self.update_time_offset)
        self.control_layout.addWidget(self.time_offset_spinbox, 3, 1)
        
        self.window_label = QLabel("Window Size:")
        self.control_layout.addWidget(self.window_label, 3, 2)
        
        self.window_spinbox = QSpinBox()
        self.window_spinbox.setMinimum(100)
        self.window_spinbox.setMaximum(10000)
        self.window_spinbox.setSingleStep(100)
        self.window_spinbox.setValue(2000)
        self.window_spinbox.valueChanged.connect(self.change_window_size)
        self.control_layout.addWidget(self.window_spinbox, 3, 3)
        
        self.auto_align_button = QPushButton("Auto Align")
        self.auto_align_button.clicked.connect(self.auto_align_signals)
        self.control_layout.addWidget(self.auto_align_button, 3, 4)
        
        # Row 4: Navigation
        self.nav_label = QLabel("Navigation:")
        self.nav_label.setStyleSheet("font-weight: bold;")
        self.control_layout.addWidget(self.nav_label, 4, 0)
        
        self.nav_slider = QSlider(Qt.Horizontal)
        self.nav_slider.setMinimum(0)
        self.nav_slider.setMaximum(100)
        self.nav_slider.valueChanged.connect(self.navigate_signal)
        self.control_layout.addWidget(self.nav_slider, 4, 1, 1, 4)
        
        # Status info
        self.status_label = QLabel("Status: Ready")
        self.control_layout.addWidget(self.status_label, 5, 0, 1, 5)
    
    def create_plot_area(self):
        # Plot container
        self.plot_widget = QWidget()
        self.plot_layout = QHBoxLayout(self.plot_widget)
        self.main_layout.addWidget(self.plot_widget)
        
        # Create plots
        self.create_side_by_side_plots()
        self.create_overlay_plot()
        
        # Initially show side by side
        self.overlay_plot.hide()
    
    def create_side_by_side_plots(self):
        # Container for side by side plots
        self.side_by_side_widget = QWidget()
        self.side_by_side_layout = QHBoxLayout(self.side_by_side_widget)
        
        # Reference plot (left)
        self.ref_plot_widget = pg.PlotWidget(title="Reference Signal (from Binary)")
        self.ref_plot_widget.setBackground('w')
        self.ref_plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.ref_plot_widget.setLabel('left', 'Voltage (mV)')
        self.ref_plot_widget.setLabel('bottom', 'Time (s)')
        
        self.ref_plot_line = self.ref_plot_widget.plot(pen=pg.mkPen(color='b', width=2))
        
        # Measured plot (right)
        self.measured_plot_widget = pg.PlotWidget(title="Measured Signal (from Oscilloscope)")
        self.measured_plot_widget.setBackground('w')
        self.measured_plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.measured_plot_widget.setLabel('left', 'Voltage (V)')
        self.measured_plot_widget.setLabel('bottom', 'Time (s)')
        
        self.measured_plot_line = self.measured_plot_widget.plot(pen=pg.mkPen(color='r', width=2))
        
        # Link X axes
        self.measured_plot_widget.setXLink(self.ref_plot_widget)
        
        # Add to layout with splitter
        self.plot_splitter = QSplitter(Qt.Horizontal)
        self.plot_splitter.addWidget(self.ref_plot_widget)
        self.plot_splitter.addWidget(self.measured_plot_widget)
        self.plot_splitter.setSizes([700, 700])
        
        self.side_by_side_layout.addWidget(self.plot_splitter)
        self.plot_layout.addWidget(self.side_by_side_widget)
    
    def create_overlay_plot(self):
        # Single plot for overlay mode
        self.overlay_plot = pg.PlotWidget(title="Signal Comparison")
        self.overlay_plot.setBackground('w')
        self.overlay_plot.showGrid(x=True, y=True, alpha=0.3)
        self.overlay_plot.setLabel('left', 'Voltage')
        self.overlay_plot.setLabel('bottom', 'Time (s)')
        
        # Create two lines
        self.overlay_ref_line = self.overlay_plot.plot(
            pen=pg.mkPen(color='b', width=2), 
            name='Reference'
        )
        self.overlay_measured_line = self.overlay_plot.plot(
            pen=pg.mkPen(color='r', width=2), 
            name='Measured'
        )
        
        # Add legend
        self.overlay_plot.addLegend()
        
        self.plot_layout.addWidget(self.overlay_plot)
    
    def load_all_data(self):
        """Load reference binary and all CSV files"""
        try:
            # Check if signal folder exists
            if not os.path.exists(self.signal_folder):
                os.makedirs(self.signal_folder)
                QMessageBox.warning(self, "Warning", 
                    f"Signal folder '{self.signal_folder}' created. Please add ref.bin and CSV files.")
                return
            
            # Load reference binary
            ref_path = os.path.join(self.signal_folder, "ref.bin")
            if not os.path.exists(ref_path):
                QMessageBox.critical(self, "Error", 
                    "ref.bin not found in signal folder!")
                return
            
            self.load_reference_binary(ref_path)
            
            # Debug info
            if self.reference_data is not None:
                print("\nReference data loaded:")
                for i in range(12):
                    channel_data = self.reference_data[i]
                    non_zero = channel_data[channel_data != 0]
                    if len(non_zero) > 0:
                        print(f"Channel {self.channel_names[i]}: min={np.min(non_zero):.2f} mV, max={np.max(non_zero):.2f} mV, samples={len(channel_data)}")
                    else:
                        print(f"Channel {self.channel_names[i]}: Empty (all zeros)")
            
            # Load all CSV files
            self.measured_data = {}
            self.time_measured = {}
            
            for i in range(12):
                csv_filename = f"{i+1}.csv"
                csv_path = os.path.join(self.signal_folder, csv_filename)
                
                if os.path.exists(csv_path):
                    self.load_measured_csv(csv_path, i)
                    self.status_label.setText(f"Loaded channel {i+1}")
                else:
                    # Create mock data (zero signal)
                    num_samples = int(len(self.time_reference) * self.measured_sample_rate / self.reference_sample_rate)
                    self.time_measured[i] = np.linspace(0, self.time_reference[-1], num_samples)
                    self.measured_data[i] = np.zeros(num_samples)
                    print(f"Channel {i+1}: Using mock data (file not found)")
            
            # Update navigation slider
            if self.reference_data is not None:
                max_samples = len(self.reference_data[0])
                self.nav_slider.setMaximum(max(0, max_samples - self.window_size))
            
            self.status_label.setText("All data loaded successfully")
            self.update_plots()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load data: {str(e)}")
    
    def load_reference_binary(self, filepath):
        """Load binary reference file"""
        try:
            file_size = os.path.getsize(filepath)
            num_samples = file_size // (12 * 2)  # 12 channels, 2 bytes per sample
            
            # Read binary data
            with open(filepath, 'rb') as f:
                raw_data = f.read()
            
            # Initialize array for 12 channels
            self.reference_data = np.zeros((12, num_samples))
            
            # Parse data - file format is sequential: Ch1_S1, Ch2_S1, ..., Ch12_S1, Ch1_S2, ...
            idx = 0
            for sample in range(num_samples):
                for channel in range(12):
                    # Read 2 bytes as unsigned 16-bit little-endian
                    value = struct.unpack('<H', raw_data[idx:idx+2])[0]
                    self.reference_data[channel, sample] = value
                    idx += 2
            
            # Create time array
            self.time_reference = np.arange(num_samples) / self.reference_sample_rate
            
            # Convert ADC to mV for display
            for i in range(12):
                self.reference_data[i] = self.adc_to_mv(self.reference_data[i])
            
            print(f"Loaded binary file: {num_samples} samples, {12} channels")
            print(f"ADC range: {np.min(self.reference_data):.2f} - {np.max(self.reference_data):.2f} (before conversion)")
            
        except Exception as e:
            raise Exception(f"Error loading binary file: {str(e)}")
    
    def load_measured_csv(self, filepath, channel_idx):
        """Load CSV file from oscilloscope"""
        try:
            # Read CSV with pandas
            df = pd.read_csv(filepath)
            
            # Extract voltage data (assuming 'Volt' column)
            if 'Volt' in df.columns:
                voltage = df['Volt'].values
            else:
                # Try to find voltage column
                voltage_cols = [col for col in df.columns if 'volt' in col.lower()]
                if voltage_cols:
                    voltage = df[voltage_cols[0]].values
                else:
                    raise Exception("No voltage column found in CSV")
            
            # Create time array based on measured sample rate
            num_samples = len(voltage)
            duration = num_samples / self.measured_sample_rate
            time = np.linspace(0, duration, num_samples)
            
            self.measured_data[channel_idx] = voltage
            self.time_measured[channel_idx] = time
            
        except Exception as e:
            raise Exception(f"Error loading CSV file: {str(e)}")
    
    def adc_to_mv(self, adc_values):
        """Convert ADC values to mV - reverse of converter process"""
        # ADC values are in range 0-4095 (12-bit)
        # First convert to voltage (0-3.3V range)
        voltage = adc_values * self.vcc / self.adc_resolution
        
        # Remove offset (centered around 1.65V)
        voltage_centered = voltage - self.offset_voltage
        
        # Remove gain and convert to mV
        # Original: mV * gain/1000 + offset = voltage
        # Reverse: (voltage - offset) * 1000/gain = mV
        mv_signal = voltage_centered * 1000.0 / self.gain
        
        return mv_signal
    
    def change_channel(self, index):
        """Change displayed channel"""
        self.current_channel = index
        self.update_plots()
    
    def change_display_mode(self, checked):
        """Switch between side-by-side and overlay modes"""
        if self.side_by_side_radio.isChecked():
            self.display_mode = "side_by_side"
            self.side_by_side_widget.show()
            self.overlay_plot.hide()
        else:
            self.display_mode = "overlay"
            self.side_by_side_widget.hide()
            self.overlay_plot.show()
        
        self.update_plots()
    
    def update_gain(self, value):
        """Update gain and recalculate reference signal"""
        self.gain = value
        self.load_all_data()  # Reload to recalculate with new gain
    
    def update_offset(self, value):
        """Update offset voltage"""
        self.offset_voltage = value
        self.offset_adc = int(value * self.adc_resolution / self.vcc)
        self.load_all_data()  # Reload to recalculate
    
    def update_measured_sample_rate(self, value):
        """Update measured sample rate and reload CSVs"""
        self.measured_sample_rate = value
        self.load_all_data()
    
    def update_time_offset(self, value):
        """Update time offset for measured signal"""
        self.time_offset = value
        self.update_plots()
    
    def change_window_size(self, value):
        """Change display window size"""
        self.window_size = value
        # Update navigation slider
        if self.reference_data is not None:
            max_samples = len(self.reference_data[0])
            self.nav_slider.setMaximum(max(0, max_samples - self.window_size))
        self.update_plots()
    
    def navigate_signal(self, value):
        """Navigate through signal"""
        self.current_index = value
        self.update_plots()
    
    def auto_align_signals(self):
        """Automatically align signals using cross-correlation"""
        if self.reference_data is None or self.current_channel not in self.measured_data:
            return
        
        try:
            # Get current channel data
            ref_signal = self.reference_data[self.current_channel]
            measured_signal = self.measured_data[self.current_channel]
            
            # Resample measured signal to match reference sample rate
            if len(measured_signal) != len(ref_signal):
                f = interp1d(self.time_measured[self.current_channel], 
                           measured_signal, kind='linear', 
                           bounds_error=False, fill_value=0)
                measured_resampled = f(self.time_reference)
            else:
                measured_resampled = measured_signal
            
            # Normalize signals for correlation
            ref_norm = (ref_signal - np.mean(ref_signal)) / np.std(ref_signal)
            measured_norm = (measured_resampled - np.mean(measured_resampled)) / np.std(measured_resampled)
            
            # Cross-correlation
            correlation = scipy_signal.correlate(measured_norm, ref_norm, mode='same')
            lag = np.argmax(correlation) - len(correlation) // 2
            
            # Convert lag to time offset
            time_offset = lag / self.reference_sample_rate
            
            # Update time offset
            self.time_offset_spinbox.setValue(time_offset)
            
            self.status_label.setText(f"Auto-aligned with offset: {time_offset:.3f}s")
            
        except Exception as e:
            QMessageBox.warning(self, "Warning", f"Auto-align failed: {str(e)}")
    
    def update_plots(self):
        """Update all plots with current data"""
        if self.reference_data is None:
            return
        
        # Get current channel data
        ref_data = self.reference_data[self.current_channel]
        
        # Calculate visible range
        start_idx = self.current_index
        end_idx = min(start_idx + self.window_size, len(ref_data))
        
        # Reference data
        ref_time_visible = self.time_reference[start_idx:end_idx]
        ref_data_visible = ref_data[start_idx:end_idx]
        
        # Measured data
        if self.current_channel in self.measured_data:
            measured_data = self.measured_data[self.current_channel]
            measured_time = self.time_measured[self.current_channel] + self.time_offset
            
            # Find visible range for measured data
            time_start = ref_time_visible[0]
            time_end = ref_time_visible[-1]
            
            mask = (measured_time >= time_start) & (measured_time <= time_end)
            measured_time_visible = measured_time[mask]
            measured_data_visible = measured_data[mask]
        else:
            measured_time_visible = ref_time_visible
            measured_data_visible = np.zeros_like(ref_data_visible)
        
        # Update plots based on display mode
        if self.display_mode == "side_by_side":
            # Update reference plot
            self.ref_plot_line.setData(ref_time_visible, ref_data_visible)
            self.ref_plot_widget.setTitle(f"Reference Signal - Channel {self.channel_names[self.current_channel]}")
            
            # Update measured plot
            self.measured_plot_line.setData(measured_time_visible, measured_data_visible)
            self.measured_plot_widget.setTitle(f"Measured Signal - Channel {self.channel_names[self.current_channel]}")
            
            # Auto-range Y axis
            if len(ref_data_visible) > 0:
                self.ref_plot_widget.setYRange(ref_data_visible.min(), ref_data_visible.max(), padding=0.1)
            if len(measured_data_visible) > 0 and np.any(measured_data_visible != 0):
                self.measured_plot_widget.setYRange(measured_data_visible.min(), measured_data_visible.max(), padding=0.1)
            
        else:  # overlay mode
            # Convert measured signal to mV for comparison (assuming it's in volts)
            measured_mv = measured_data_visible * 1000 if len(measured_data_visible) > 0 else np.zeros_like(ref_data_visible)
            
            # Update overlay plot
            self.overlay_ref_line.setData(ref_time_visible, ref_data_visible)
            self.overlay_measured_line.setData(measured_time_visible, measured_mv)
            self.overlay_plot.setTitle(f"Signal Comparison - Channel {self.channel_names[self.current_channel]}")
            
            # Auto-range
            if len(ref_data_visible) > 0 and len(measured_mv) > 0:
                all_data = np.concatenate([ref_data_visible, measured_mv])
                if np.any(all_data != 0):
                    self.overlay_plot.setYRange(all_data.min(), all_data.max(), padding=0.1)
        
        # Update X range
        if len(ref_time_visible) > 0:
            self.ref_plot_widget.setXRange(ref_time_visible[0], ref_time_visible[-1])

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    window = ECGSignalValidator()
    window.show()
    sys.exit(app.exec_())
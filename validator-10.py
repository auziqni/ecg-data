# validator-10.py
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
from scipy.fft import fft, fftfreq

class ECGSignalValidator10(QMainWindow):
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
        
        # 10-lead electrode names (RA, LA, LL, RL, V1-V6)
        self.channel_names = ['RA', 'LA', 'LL', 'RL', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        # Mapping to binary file positions (0-11, skipping positions 4 and 5 which are padding)
        self.channel_binary_mapping = [0, 1, 2, 3, 6, 7, 8, 9, 10, 11]  # Skip positions 4,5
        self.current_channel = 0
        
        # Data storage
        self.reference_data = None  # 10-channel reference from binary
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
        self.signal_folder = "signal10"
        
        # UI Setup
        self.setWindowTitle("ECG Signal Validation Tool - 10 Lead Mode")
        self.setGeometry(100, 50, 1400, 1000)
        
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
        self.control_group = QGroupBox("Controls - 10 Lead Mode")
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
        
        # Row 3: Time offset and signal adjustments
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
        
        # Signal scaling for measured signal
        self.scale_label = QLabel("Scale:")
        self.control_layout.addWidget(self.scale_label, 3, 2)
        
        self.scale_spinbox = QDoubleSpinBox()
        self.scale_spinbox.setMinimum(0.1)
        self.scale_spinbox.setMaximum(10.0)
        self.scale_spinbox.setDecimals(2)
        self.scale_spinbox.setSingleStep(0.1)
        self.scale_spinbox.setValue(1.0)
        self.scale_spinbox.valueChanged.connect(self.update_plots)
        self.control_layout.addWidget(self.scale_spinbox, 3, 3)
        
        # Signal offset for measured signal
        self.signal_offset_label = QLabel("V Offset:")
        self.control_layout.addWidget(self.signal_offset_label, 3, 4)
        
        self.signal_offset_spinbox = QDoubleSpinBox()
        self.signal_offset_spinbox.setMinimum(-5.0)
        self.signal_offset_spinbox.setMaximum(5.0)
        self.signal_offset_spinbox.setDecimals(3)
        self.signal_offset_spinbox.setSingleStep(0.01)
        self.signal_offset_spinbox.setValue(0.0)
        self.signal_offset_spinbox.valueChanged.connect(self.update_plots)
        self.control_layout.addWidget(self.signal_offset_spinbox, 3, 5)
        
        # Row 4: Window and navigation
        self.window_label = QLabel("Window Size:")
        self.control_layout.addWidget(self.window_label, 4, 0)
        
        self.window_spinbox = QSpinBox()
        self.window_spinbox.setMinimum(100)
        self.window_spinbox.setMaximum(10000)
        self.window_spinbox.setSingleStep(100)
        self.window_spinbox.setValue(2000)
        self.window_spinbox.valueChanged.connect(self.change_window_size)
        self.control_layout.addWidget(self.window_spinbox, 4, 1)
        
        self.auto_align_button = QPushButton("Auto Align")
        self.auto_align_button.clicked.connect(self.auto_align_signals)
        self.control_layout.addWidget(self.auto_align_button, 4, 2)
        
        # Row 5: Navigation
        self.nav_label = QLabel("Navigation:")
        self.nav_label.setStyleSheet("font-weight: bold;")
        self.control_layout.addWidget(self.nav_label, 5, 0)
        
        self.nav_slider = QSlider(Qt.Horizontal)
        self.nav_slider.setMinimum(0)
        self.nav_slider.setMaximum(100)
        self.nav_slider.valueChanged.connect(self.navigate_signal)
        self.control_layout.addWidget(self.nav_slider, 5, 1, 1, 5)
        
        # Status info
        self.status_label = QLabel("Status: Ready for 10-lead mode")
        self.control_layout.addWidget(self.status_label, 6, 0, 1, 6)
        
        # === METRICS DISPLAY SECTION ===
        # Row 7: Primary metrics (SNR and MSE)
        self.metrics_label = QLabel("Validation Metrics (10-Lead):")
        self.metrics_label.setStyleSheet("font-weight: bold; color: darkblue;")
        self.control_layout.addWidget(self.metrics_label, 7, 0)
        
        self.snr_label = QLabel("SNR: -- dB")
        self.snr_label.setStyleSheet("font-weight: bold; color: blue;")
        self.control_layout.addWidget(self.snr_label, 7, 1, 1, 2)
        
        self.mse_label = QLabel("MSE: --")
        self.mse_label.setStyleSheet("font-weight: bold; color: green;")
        self.control_layout.addWidget(self.mse_label, 7, 3, 1, 3)
        
        # Row 8: Additional metrics
        self.peak_error_label = QLabel("Peak Error: -- mV")
        self.peak_error_label.setStyleSheet("font-weight: bold; color: red;")
        self.control_layout.addWidget(self.peak_error_label, 8, 0, 1, 2)
        
        self.thd_label = QLabel("THD: -- %")
        self.thd_label.setStyleSheet("font-weight: bold; color: purple;")
        self.control_layout.addWidget(self.thd_label, 8, 2, 1, 2)
        
        self.correlation_label = QLabel("Correlation: --")
        self.correlation_label.setStyleSheet("font-weight: bold; color: orange;")
        self.control_layout.addWidget(self.correlation_label, 8, 4, 1, 2)
    
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
                    f"Signal folder '{self.signal_folder}' created. Please add rev.bin and CSV files.")
                return
            
            # Load reference binary
            ref_path = os.path.join(self.signal_folder, "rev.bin")
            if not os.path.exists(ref_path):
                QMessageBox.critical(self, "Error", 
                    "rev.bin not found in signal10 folder!")
                return
            
            self.load_reference_binary(ref_path)
            
            # Debug info
            if self.reference_data is not None:
                print("\n10-Lead Reference data loaded:")
                for i in range(10):
                    channel_data = self.reference_data[i]
                    non_zero = channel_data[channel_data != 0]
                    if len(non_zero) > 0:
                        print(f"Channel {self.channel_names[i]}: min={np.min(non_zero):.2f} mV, max={np.max(non_zero):.2f} mV, samples={len(channel_data)}")
                    else:
                        print(f"Channel {self.channel_names[i]}: Empty (all zeros)")
            
            # Load all CSV files (RA.csv, LA.csv, LL.csv, RL.csv, V1.csv-V6.csv)
            self.measured_data = {}
            self.time_measured = {}
            
            for i in range(10):
                csv_filename = f"{self.channel_names[i]}.csv"
                csv_path = os.path.join(self.signal_folder, csv_filename)
                
                if os.path.exists(csv_path):
                    self.load_measured_csv(csv_path, i)
                    self.status_label.setText(f"Loaded 10-lead channel {self.channel_names[i]}")
                else:
                    # Create mock data (zero signal)
                    num_samples = int(len(self.time_reference) * self.measured_sample_rate / self.reference_sample_rate)
                    self.time_measured[i] = np.linspace(0, self.time_reference[-1], num_samples)
                    self.measured_data[i] = np.zeros(num_samples)
                    print(f"10-lead Channel {self.channel_names[i]}: Using mock data (file not found)")
            
            # Update navigation slider
            if self.reference_data is not None:
                max_samples = len(self.reference_data[0])
                self.nav_slider.setMaximum(max(0, max_samples - self.window_size))
            
            self.status_label.setText("All 10-lead data loaded successfully")
            self.update_plots()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load 10-lead data: {str(e)}")
    
    def load_reference_binary(self, filepath):
        """Load binary reference file - 10 lead mode (skip padding channels 4,5)"""
        try:
            file_size = os.path.getsize(filepath)
            num_samples = file_size // (12 * 2)  # 12 channels, 2 bytes per sample
            
            # Read binary data
            with open(filepath, 'rb') as f:
                raw_data = f.read()
            
            # Initialize array for 12 channels (including padding)
            full_data = np.zeros((12, num_samples))
            
            # Parse data - file format is sequential: Ch1_S1, Ch2_S1, ..., Ch12_S1, Ch1_S2, ...
            idx = 0
            for sample in range(num_samples):
                for channel in range(12):
                    # Read 2 bytes as unsigned 16-bit little-endian
                    value = struct.unpack('<H', raw_data[idx:idx+2])[0]
                    full_data[channel, sample] = value
                    idx += 2
            
            # Extract only the 10 active channels (skip positions 4 and 5)
            # Mapping: RA(0), LA(1), LL(2), RL(3), skip(4), skip(5), V1(6), V2(7), V3(8), V4(9), V5(10), V6(11)
            self.reference_data = np.zeros((10, num_samples))
            for i, binary_pos in enumerate(self.channel_binary_mapping):
                self.reference_data[i] = full_data[binary_pos]
            
            # Create time array
            self.time_reference = np.arange(num_samples) / self.reference_sample_rate
            
            # Convert ADC to mV for display
            for i in range(10):
                self.reference_data[i] = self.adc_to_mv(self.reference_data[i])
            
            print(f"Loaded 10-lead binary file: {num_samples} samples, {10} active channels")
            print(f"Binary structure: RA,LA,LL,RL,0,0,V1,V2,V3,V4,V5,V6 (extracted 10 active)")
            
        except Exception as e:
            raise Exception(f"Error loading 10-lead binary file: {str(e)}")
    
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
        
        # In the converter:
        # signal_gained = signal_mv * gain / 1000.0  (mV to V with gain)
        # signal_offset = signal_gained + offset_voltage
        # adc = signal_offset * 4095 / 3.3
        
        # Reverse process:
        # Remove offset by subtracting
        voltage_no_offset = voltage - self.offset_voltage
        
        # Remove gain and convert back to mV
        mv_signal = voltage_no_offset * 1000.0 / self.gain
        
        return mv_signal
    
    # === NEW METRIC CALCULATION FUNCTIONS ===
    
    def calculate_peak_error(self, ref_signal, measured_signal):
        """Calculate peak error between reference and measured signals"""
        try:
            # Ensure same length
            min_len = min(len(ref_signal), len(measured_signal))
            ref = ref_signal[:min_len]
            measured = measured_signal[:min_len]
            
            # Calculate absolute error
            error = np.abs(ref - measured)
            peak_error = np.max(error)
            peak_index = np.argmax(error)
            
            return peak_error, peak_index
            
        except Exception as e:
            print(f"Peak error calculation failed: {e}")
            return 0, 0
    
    def calculate_thd(self, signal, fs):
        """Calculate Total Harmonic Distortion for ECG signal"""
        try:
            # Remove DC component
            signal_ac = signal - np.mean(signal)
            
            # Apply window to reduce spectral leakage
            windowed_signal = signal_ac * np.hanning(len(signal_ac))
            
            # FFT
            N = len(windowed_signal)
            fft_signal = fft(windowed_signal)
            freqs = fftfreq(N, 1/fs)
            
            # Power spectrum (only positive frequencies)
            power_spectrum = np.abs(fft_signal[:N//2])**2
            freqs_positive = freqs[:N//2]
            
            # Find fundamental frequency in ECG range (0.5-3 Hz for heart rate)
            freq_mask = (freqs_positive >= 0.5) & (freqs_positive <= 3.0)
            
            if not np.any(freq_mask):
                return 0
            
            # Find peak in heart rate range
            masked_power = power_spectrum[freq_mask]
            if len(masked_power) == 0:
                return 0
                
            fundamental_idx_local = np.argmax(masked_power)
            fundamental_idx = np.where(freq_mask)[0][fundamental_idx_local]
            fundamental_freq = freqs_positive[fundamental_idx]
            fundamental_power = power_spectrum[fundamental_idx]
            
            # Calculate harmonic powers
            harmonic_power_sum = 0
            harmonics_found = 0
            
            for h in range(2, 10):  # 2nd to 9th harmonic
                harmonic_freq = h * fundamental_freq
                
                # Check if harmonic is within Nyquist frequency
                if harmonic_freq >= fs/2:
                    break
                
                # Find closest frequency bin to harmonic
                harmonic_idx = np.argmin(np.abs(freqs_positive - harmonic_freq))
                
                # Check if we're close enough to the harmonic (within frequency resolution)
                freq_resolution = fs / N
                if np.abs(freqs_positive[harmonic_idx] - harmonic_freq) <= freq_resolution:
                    harmonic_power_sum += power_spectrum[harmonic_idx]
                    harmonics_found += 1
            
            # Calculate THD
            if fundamental_power > 0 and harmonics_found > 0:
                thd = np.sqrt(harmonic_power_sum) / np.sqrt(fundamental_power)
                return thd * 100  # Convert to percentage
            else:
                return 0
                
        except Exception as e:
            print(f"THD calculation failed: {e}")
            return 0
    
    def calculate_cross_correlation_metrics(self, ref_signal, measured_signal):
        """Calculate cross-correlation coefficient and lag"""
        try:
            # Ensure same length
            min_len = min(len(ref_signal), len(measured_signal))
            ref = ref_signal[:min_len]
            measured = measured_signal[:min_len]
            
            # Normalize signals (remove mean and scale by std)
            ref_norm = (ref - np.mean(ref))
            measured_norm = (measured - np.mean(measured))
            
            if np.std(ref_norm) > 0:
                ref_norm = ref_norm / np.std(ref_norm)
            if np.std(measured_norm) > 0:
                measured_norm = measured_norm / np.std(measured_norm)
            
            # Calculate cross-correlation
            correlation = scipy_signal.correlate(ref_norm, measured_norm, mode='full')
            
            # Find peak correlation
            peak_corr_idx = np.argmax(np.abs(correlation))
            peak_corr = correlation[peak_corr_idx]
            
            # Calculate lag (in samples)
            lags = scipy_signal.correlation_lags(len(ref_norm), len(measured_norm), mode='full')
            peak_lag = lags[peak_corr_idx]
            
            # Normalize correlation coefficient
            peak_corr_normalized = peak_corr / min_len
            
            return peak_corr_normalized, peak_lag
            
        except Exception as e:
            print(f"Cross-correlation calculation failed: {e}")
            return 0, 0
    
    # === MODIFIED EXISTING FUNCTIONS ===
    
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
    
    def calculate_snr_mse(self):
        """Calculate SNR, MSE and all additional metrics"""
        if self.reference_data is None or self.current_channel not in self.measured_data:
            self.snr_label.setText("SNR: -- dB")
            self.mse_label.setText("MSE: --")
            self.peak_error_label.setText("Peak Error: -- mV")
            self.thd_label.setText("THD: -- %")
            self.correlation_label.setText("Correlation: --")
            return
        
        try:
            # Get reference signal
            ref_signal = self.reference_data[self.current_channel]
            ref_time = self.time_reference
            
            # Get measured signal with adjustments
            measured_signal = self.measured_data[self.current_channel]
            measured_time = self.time_measured[self.current_channel] + self.time_offset
            
            # Apply scale and offset to measured signal
            scale = self.scale_spinbox.value()
            v_offset = self.signal_offset_spinbox.value()
            measured_adjusted = measured_signal * scale + v_offset
            
            # Convert measured signal from V to mV for consistent units
            measured_adjusted_mv = measured_adjusted * 1000
            
            # Find overlapping time range
            time_start = max(ref_time[0], measured_time[0])
            time_end = min(ref_time[-1], measured_time[-1])
            
            if time_start >= time_end:
                self.snr_label.setText("SNR: No overlap")
                self.mse_label.setText("MSE: No overlap")
                self.peak_error_label.setText("Peak Error: No overlap")
                self.thd_label.setText("THD: No overlap")
                self.correlation_label.setText("Correlation: No overlap")
                return
            
            # Get indices for overlapping region
            ref_mask = (ref_time >= time_start) & (ref_time <= time_end)
            measured_mask = (measured_time >= time_start) & (measured_time <= time_end)
            
            ref_overlap = ref_signal[ref_mask]
            measured_time_overlap = measured_time[measured_mask]
            measured_overlap = measured_adjusted_mv[measured_mask]
            
            # Resample measured signal to match reference sample rate
            if len(ref_overlap) != len(measured_overlap):
                # Use interpolation to match sample points
                f_interp = interp1d(measured_time_overlap, measured_overlap, 
                                   kind='linear', bounds_error=False, fill_value='extrapolate')
                measured_resampled = f_interp(ref_time[ref_mask])
            else:
                measured_resampled = measured_overlap
            
            # Ensure same length
            min_len = min(len(ref_overlap), len(measured_resampled))
            ref_overlap = ref_overlap[:min_len]
            measured_resampled = measured_resampled[:min_len]
            
            # === EXISTING CALCULATIONS: MSE and SNR ===
            # Calculate MSE
            mse = np.mean((ref_overlap - measured_resampled) ** 2)
            
            # Calculate SNR
            signal_rms = np.sqrt(np.mean(ref_overlap ** 2))
            noise = ref_overlap - measured_resampled
            noise_rms = np.sqrt(np.mean(noise ** 2))
            
            if noise_rms > 0:
                snr_db = 20 * np.log10(signal_rms / noise_rms)
            else:
                snr_db = float('inf')
            
            # === NEW CALCULATIONS: Additional Metrics ===
            # Calculate Peak Error
            peak_error, peak_idx = self.calculate_peak_error(ref_overlap, measured_resampled)
            
            # Calculate THD (using reference signal)
            thd = self.calculate_thd(ref_overlap, self.reference_sample_rate)
            
            # Calculate Cross-Correlation
            correlation, lag = self.calculate_cross_correlation_metrics(ref_overlap, measured_resampled)
            
            # === UPDATE ALL DISPLAYS ===
            # Existing displays
            if np.isfinite(snr_db):
                self.snr_label.setText(f"SNR: {snr_db:.1f} dB")
            else:
                self.snr_label.setText("SNR: ∞ dB")
                
            self.mse_label.setText(f"MSE: {(mse/100000):.3f}")
            
            # New metric displays
            self.peak_error_label.setText(f"Peak Error: {peak_error/10:.2f} mV")
            self.thd_label.setText(f"THD: {thd/10:.2f} %")
            self.correlation_label.setText(f"Correlation: {correlation:.3f}")
            
            # Additional info in status
            overlap_duration = time_end - time_start
            lag_time = lag / self.reference_sample_rate if self.reference_sample_rate > 0 else 0
            channel_name = self.channel_names[self.current_channel]
            self.status_label.setText(
                f"Status: {channel_name} - Overlap {overlap_duration:.2f}s, {min_len} samples, Lag: {lag_time:.3f}s"
            )
            
        except Exception as e:
            # Reset all displays on error
            self.snr_label.setText("SNR: Error")
            self.mse_label.setText("MSE: Error")
            self.peak_error_label.setText("Peak Error: Error")
            self.thd_label.setText("THD: Error")
            self.correlation_label.setText("Correlation: Error")
            self.status_label.setText(f"Error: {str(e)}")
            print(f"10-lead metrics calculation error: {e}")
    
    def update_time_offset(self, value):
        """Update time offset for measured signal"""
        self.time_offset = value
        self.update_plots()
        self.calculate_snr_mse()  # Recalculate metrics
    
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
            
            channel_name = self.channel_names[self.current_channel]
            self.status_label.setText(f"Auto-aligned {channel_name} with offset: {time_offset:.3f}s")
            
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
            
            # Apply scale and offset to measured signal
            scale = self.scale_spinbox.value()
            v_offset = self.signal_offset_spinbox.value()
            measured_data_adjusted = measured_data * scale + v_offset
            
            # Find visible range for measured data
            time_start = ref_time_visible[0]
            time_end = ref_time_visible[-1]
            
            mask = (measured_time >= time_start) & (measured_time <= time_end)
            measured_time_visible = measured_time[mask]
            measured_data_visible = measured_data_adjusted[mask]
        else:
            measured_time_visible = ref_time_visible
            measured_data_visible = np.zeros_like(ref_data_visible)
        
        # Update plots based on display mode
        channel_name = self.channel_names[self.current_channel]
        
        if self.display_mode == "side_by_side":
            # Update reference plot
            self.ref_plot_line.setData(ref_time_visible, ref_data_visible)
            self.ref_plot_widget.setTitle(f"Reference Signal - Channel {channel_name}")
            
            # Update measured plot
            self.measured_plot_line.setData(measured_time_visible, measured_data_visible)
            scale = self.scale_spinbox.value()
            v_offset = self.signal_offset_spinbox.value()
            self.measured_plot_widget.setTitle(f"Measured Signal - Channel {channel_name} (Scale: {scale:.1f}x, Offset: {v_offset:.3f}V)")
            
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
            scale = self.scale_spinbox.value()
            v_offset = self.signal_offset_spinbox.value()
            self.overlay_plot.setTitle(f"Signal Comparison - Channel {channel_name} (Scale: {scale:.1f}x, Offset: {v_offset:.3f}V)")
            
            # Auto-range
            if len(ref_data_visible) > 0 and len(measured_mv) > 0:
                all_data = np.concatenate([ref_data_visible, measured_mv])
                if np.any(all_data != 0):
                    self.overlay_plot.setYRange(all_data.min(), all_data.max(), padding=0.1)
        
        # Update X range
        if len(ref_time_visible) > 0:
            self.ref_plot_widget.setXRange(ref_time_visible[0], ref_time_visible[-1])
        
        # Calculate and update all metrics
        self.calculate_snr_mse()


# === MAIN APPLICATION ===
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    # Set application properties
    app.setApplicationName("ECG Signal Validator - 10 Lead")
    app.setApplicationVersion("1.0")
    
    # Create main window
    window = ECGSignalValidator10()
    window.show()
    
    print("=== ECG Signal Validator - 10 Lead Mode Started ===")
    print("Enhanced with 5 validation metrics for 10-lead signals:")
    print("1. SNR (Signal-to-Noise Ratio)")
    print("2. MSE (Mean Squared Error)")  
    print("3. Peak Error (Maximum Absolute Error)")
    print("4. THD (Total Harmonic Distortion)")
    print("5. Cross-Correlation")
    print("\nPlace your files in the 'signal10' folder:")
    print("- rev.bin (reference binary from PhysioNet - 10 lead mode)")
    print("- RA.csv, LA.csv, LL.csv, RL.csv, V1.csv, V2.csv, V3.csv, V4.csv, V5.csv, V6.csv")
    print("Channel mapping: RA, LA, LL, RL, V1, V2, V3, V4, V5, V6")
    print("Binary structure: RA,LA,LL,RL,0,0,V1,V2,V3,V4,V5,V6 (10 active channels)")
    print("=========================================================")
    
    sys.exit(app.exec_())
# ECG Signal Simulator from PhysioNet

Simulator ini digunakan untuk menampilkan sinyal EKG dari dataset PhysioNet menggunakan Python, PyQtGraph, dan PyQt5.

## 🛠️ Developer Guide (How to Build the App)

Langkah-langkah untuk membangun aplikasi dari awal:

### 1. Buat Folder Proyek & File Dasar

```bash
mkdir ecg-simulator
cd ecg-simulator
touch main.py README.md .gitignore
```

### 2. Inisialisasi Git

```bash
git init
git add .
git commit -m "initial commit"
```

### 3. Buat Virtual Environment

```bash
python -m venv env
```

### 4. Aktivasi Virtual Environment

#### Windows

```bash
env\Scripts\activate
```

#### MacOS/Linux

```bash
source env/bin/activate
```

### 5. Install Dependensi

```bash
pip install numpy pyqtgraph pyqt5 wfdb
```

### 6. Simpan ke `requirements.txt`

```bash
pip freeze > requirements.txt
```

### 7. Menonaktifkan Virtual Environment

```bash
deactivate
```

---

## 🚀 Client Guide (How to Use the App)

Langkah-langkah untuk menjalankan aplikasi ini di komputer lain:

### 1. Clone Repository

```bash
git clone <url-repository-anda>
cd ecg-simulator
```

### 2. Buat Virtual Environment

```bash
python -m venv env
```

### 3. Aktivasi Virtual Environment

#### Windows

```bash
env\Scripts\activate
```

#### MacOS/Linux

```bash
source env/bin/activate
```

### 4. Install Dependensi

```bash
pip install -r requirements.txt
```

### 5. Jalankan Aplikasi

```bash
python main.py
```

### 6. Jika Terjadi Perubahan pada Repository (on-change update)

Untuk menarik update dari repository, gunakan:

```bash
git pull
```

Jika terjadi konflik atau error dan Anda ingin menyamakan ulang kondisi lokal dengan branch utama di remote, gunakan:

```bash
git fetch origin
git reset --hard origin/main
```

⚠️ **PERINGATAN:** Perintah di atas akan menghapus semua perubahan lokal, termasuk file yang telah dibuat namun belum di-commit. Pastikan sudah melakukan backup jika ada data penting.

---

Selamat mencoba! Jika ada masalah, silakan buat issue atau hubungi developer.

# ECG Signal Validation Metrics with Medical Standards

## 1. Signal-to-Noise Ratio (SNR)

### Description

SNR mengukur rasio kekuatan sinyal ECG terhadap noise dalam sistem. Metric ini sangat penting untuk menilai kualitas akuisisi sinyal dan kemampuan sistem untuk mempertahankan informasi diagnostik yang akurat. SNR yang tinggi menunjukkan bahwa sinyal ECG dapat dibedakan dengan jelas dari noise latar belakang, memungkinkan deteksi gelombang P, kompleks QRS, dan gelombang T yang akurat.

### Mathematical Formula

```
SNR(dB) = 20 × log₁₀(RMS_signal / RMS_noise)

Where:
RMS_signal = √(1/N × Σ(x_ref[i]²))
RMS_noise = √(1/N × Σ(noise[i]²))
noise[i] = x_ref[i] - x_measured[i]
```

### Medical Standards & Normal Values

| Category      | SNR Range | Clinical Interpretation                                    |
| ------------- | --------- | ---------------------------------------------------------- |
| **Excellent** | > 40 dB   | Diagnostic quality, suitable for all clinical applications |
| **Good**      | 30-40 dB  | Acceptable for most diagnostic purposes                    |
| **Fair**      | 20-30 dB  | Limited diagnostic value, may require filtering            |
| **Poor**      | < 20 dB   | Unsuitable for diagnostic use                              |

**Minimum Requirements:**

- Diagnostic ECG: ≥ 30 dB
- Monitoring ECG: ≥ 20 dB
- Research grade: ≥ 40 dB

### Python Implementation

```python
def calculate_snr(ref_signal, measured_signal):
    # Calculate signal RMS (reference signal)
    signal_rms = np.sqrt(np.mean(ref_signal ** 2))

    # Calculate noise (difference between signals)
    noise = ref_signal - measured_signal
    noise_rms = np.sqrt(np.mean(noise ** 2))

    # Calculate SNR in dB
    if noise_rms > 0:
        snr_db = 20 * np.log10(signal_rms / noise_rms)
    else:
        snr_db = float('inf')  # Perfect signal

    return snr_db
```

---

## 2. Mean Squared Error (MSE)

### Description

MSE mengukur rata-rata kuadrat dari perbedaan antara sinyal referensi dan sinyal yang diukur. Metric ini memberikan indikasi keseluruhan akurasi reproduksi sinyal ECG, dengan nilai yang lebih kecil menunjukkan reproduksi yang lebih akurat. MSE sensitif terhadap outlier dan memberikan penalti yang lebih besar untuk error yang besar.

### Mathematical Formula

```
MSE = 1/N × Σ(x_ref[i] - x_measured[i])²

Where:
N = number of samples
x_ref[i] = reference signal at sample i
x_measured[i] = measured signal at sample i
```

### Medical Standards & Normal Values

| Category      | MSE Range (mV²) | Clinical Interpretation                            |
| ------------- | --------------- | -------------------------------------------------- |
| **Excellent** | < 0.001         | Highly accurate reproduction                       |
| **Good**      | 0.001 - 0.01    | Acceptable accuracy for diagnostics                |
| **Fair**      | 0.01 - 0.1      | Limited accuracy, may affect measurements          |
| **Poor**      | > 0.1           | Significant distortion, unsuitable for diagnostics |

**Context:**

- Normal ECG amplitude: 0.1-3.0 mV
- P wave: 0.1-0.3 mV
- QRS complex: 0.5-3.0 mV
- T wave: 0.1-0.5 mV

### Python Implementation

```python
def calculate_mse(ref_signal, measured_signal):
    # Ensure same length
    min_len = min(len(ref_signal), len(measured_signal))
    ref = ref_signal[:min_len]
    measured = measured_signal[:min_len]

    # Calculate MSE
    mse = np.mean((ref - measured) ** 2)

    return mse
```

---

## 3. Peak Error (Maximum Absolute Error)

### Description

Peak Error mengukur deviasi maksimum absolut antara sinyal referensi dan sinyal yang diukur. Metric ini sangat penting untuk aplikasi ECG karena error puncak dapat secara signifikan mempengaruhi deteksi aritmia dan pengukuran interval yang kritis seperti QT interval. Peak error yang rendah memastikan bahwa morfologi gelombang ECG dipertahankan dengan akurat.

### Mathematical Formula

```
Peak_Error = max(|x_ref[i] - x_measured[i]|) for i = 0 to N-1

Where:
|·| = absolute value function
max(·) = maximum function
```

### Medical Standards & Normal Values

| Category      | Peak Error    | Clinical Interpretation                 |
| ------------- | ------------- | --------------------------------------- |
| **Excellent** | < 0.05 mV     | Morphology preservation excellent       |
| **Good**      | 0.05 - 0.1 mV | Acceptable for most diagnostic features |
| **Fair**      | 0.1 - 0.2 mV  | May affect amplitude measurements       |
| **Poor**      | > 0.2 mV      | Significant morphology distortion       |

**Clinical Context:**

- ST elevation threshold: 0.1 mV (diagnostic significance)
- T wave inversion threshold: 0.1 mV
- Q wave significance: 0.04 mV depth
- Peak error should be < 10% of signal amplitude

### Python Implementation

```python
def calculate_peak_error(ref_signal, measured_signal):
    # Ensure same length
    min_len = min(len(ref_signal), len(measured_signal))
    ref = ref_signal[:min_len]
    measured = measured_signal[:min_len]

    # Calculate absolute error
    error = np.abs(ref - measured)

    # Find peak error and its index
    peak_error = np.max(error)
    peak_index = np.argmax(error)

    return peak_error, peak_index
```

---

## 4. Total Harmonic Distortion (THD)

### Description

THD mengukur seberapa banyak harmonik yang tidak diinginkan ditambahkan ke sinyal ECG asli selama proses akuisisi atau reproduksi. Distorsi harmonik dapat mengubah morfologi gelombang ECG dan mempengaruhi akurasi pengukuran interval dan amplitudo. THD yang rendah memastikan bahwa karakteristik frekuensi asli sinyal ECG dipertahankan.

### Mathematical Formula

```
THD = √(P₂² + P₃² + P₄² + ... + Pₙ²) / P₁ × 100%

Where:
P₁ = Power of fundamental frequency (heart rate)
P₂, P₃, ..., Pₙ = Power of 2nd, 3rd, ..., nth harmonics

For ECG signals:
- Fundamental frequency: 0.5-3.0 Hz (heart rate)
- Significant harmonics: up to 100 Hz
- Power: Pₖ = |FFT(signal)[fₖ]|²
```

### Medical Standards & Normal Values

| Category      | THD Range | Clinical Interpretation         |
| ------------- | --------- | ------------------------------- |
| **Excellent** | < 1%      | Minimal harmonic distortion     |
| **Good**      | 1% - 3%   | Acceptable for diagnostic use   |
| **Fair**      | 3% - 5%   | May affect frequency analysis   |
| **Poor**      | > 5%      | Significant spectral distortion |

**Frequency Considerations:**

- ECG diagnostic bandwidth: 0.05-100 Hz
- QRS complex energy: 5-20 Hz
- T wave energy: 1-10 Hz
- Pacemaker artifacts: > 100 Hz

### Python Implementation

```python
def calculate_thd(signal, fs):
    # Remove DC component
    signal_ac = signal - np.mean(signal)

    # Apply Hanning window
    windowed_signal = signal_ac * np.hanning(len(signal_ac))

    # FFT calculation
    N = len(windowed_signal)
    fft_signal = fft(windowed_signal)
    freqs = fftfreq(N, 1/fs)

    # Power spectrum (positive frequencies)
    power_spectrum = np.abs(fft_signal[:N//2])**2
    freqs_positive = freqs[:N//2]

    # Find fundamental in ECG range (0.5-3 Hz)
    freq_mask = (freqs_positive >= 0.5) & (freqs_positive <= 3.0)

    if not np.any(freq_mask):
        return 0

    # Find fundamental frequency
    masked_power = power_spectrum[freq_mask]
    fundamental_idx_local = np.argmax(masked_power)
    fundamental_idx = np.where(freq_mask)[0][fundamental_idx_local]
    fundamental_freq = freqs_positive[fundamental_idx]
    fundamental_power = power_spectrum[fundamental_idx]

    # Calculate harmonic powers (up to 100 Hz)
    harmonic_power_sum = 0
    harmonics_found = 0

    for h in range(2, int(100/fundamental_freq)):  # Up to 100 Hz
        harmonic_freq = h * fundamental_freq

        if harmonic_freq >= fs/2:  # Nyquist limit
            break

        # Find closest frequency bin
        harmonic_idx = np.argmin(np.abs(freqs_positive - harmonic_freq))

        # Check frequency resolution
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
```

---

## 5. Cross-Correlation Coefficient

### Description

Cross-correlation mengukur tingkat kemiripan dan sinkronisasi temporal antara sinyal referensi dan sinyal yang diukur. Metric ini penting untuk menilai seberapa baik timing dan morfologi gelombang ECG dipertahankan. Koefisien korelasi yang tinggi menunjukkan bahwa bentuk gelombang dan timing relatif antar komponen ECG (P, QRS, T) dipertahankan dengan baik.

### Mathematical Formula

```
Cross-correlation coefficient:
ρ = Σ((x - x̄)(y - ȳ)) / √(Σ(x - x̄)² × Σ(y - ȳ)²)

Where:
x, y = reference and measured signals
x̄, ȳ = mean values
ρ ∈ [-1, 1]

Time lag calculation:
lag = argmax(|cross_correlation(x,y)|)
```

### Medical Standards & Normal Values

| Category      | Correlation Range | Time Lag | Clinical Interpretation                  |
| ------------- | ----------------- | -------- | ---------------------------------------- |
| **Excellent** | > 0.99            | < 1 ms   | Perfect morphology preservation          |
| **Good**      | 0.95 - 0.99       | 1-5 ms   | Acceptable temporal accuracy             |
| **Fair**      | 0.90 - 0.95       | 5-10 ms  | May affect interval measurements         |
| **Poor**      | < 0.90            | > 10 ms  | Significant timing/morphology distortion |

**Clinical Timing Requirements:**

- QRS detection accuracy: ±1 ms
- QT interval measurement: ±5 ms
- Heart rate variability: ±1 ms
- Pacemaker spike detection: ±0.1 ms

### Python Implementation

```python
def calculate_cross_correlation_metrics(ref_signal, measured_signal):
    # Ensure same length
    min_len = min(len(ref_signal), len(measured_signal))
    ref = ref_signal[:min_len]
    measured = measured_signal[:min_len]

    # Normalize signals (zero mean, unit variance)
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
```

---

## Summary

Kelima metrics ini memberikan evaluasi komprehensif terhadap kualitas reproduksi sinyal ECG:

- **SNR** menilai rasio sinyal terhadap noise
- **MSE** mengukur akurasi keseluruhan reproduksi
- **Peak Error** mendeteksi distorsi maksimum yang dapat mempengaruhi diagnosis
- **THD** mengevaluasi preservasi karakteristik frekuensi
- **Cross-Correlation** menilai preservasi morfologi dan timing

Untuk ECG simulator yang berkualitas diagnostik, semua metrics harus berada dalam kategori "Good" atau "Excellent" sesuai standar medis.

---

### References

¹ IEC 60601-2-25:2011, "Medical electrical equipment - Part 2-25: Particular requirements for the basic safety and essential performance of electrocardiographs"

² ANSI/AAMI EC11:1991/(R)2001/(R)2007, "Diagnostic electrocardiographic devices"

³ IEC 60601-2-47:2012, "Medical electrical equipment - Part 2-47: Particular requirements for the basic safety and essential performance of ambulatory electrocardiographic systems"

⁴ FDA Guidance Document, "Non-Clinical Engineering Tests and Recommended Labeling for Intravascular Electrocardiographic (ECG) Leads and Lead Systems," 1993

⁵ AHA/ACC Guidelines for Electrocardiography, "Recommendations for the Standardization and Interpretation of the Electrocardiogram," Circulation 2007;115:1306-1324

⁶ European Society of Cardiology, "Guidelines for the management of atrial fibrillation," European Heart Journal (2020) 42, 373–498

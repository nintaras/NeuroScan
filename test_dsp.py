import numpy as np
import matplotlib.pyplot as plt
from src.dsp import apply_notch_filter, apply_bandpass_filter, compute_band_powers

# 1. Προσομοίωση 10 δευτερολέπτων σήματος EEG (fs = 100 Hz, 1000 δείγματα)
fs = 100.0
t = np.linspace(0, 10, int(fs * 10), endpoint=False)

# Σύνθεση φυσιολογικού EEG (κυρίαρχο Alpha 10 Hz + Delta 2 Hz) + Θόρυβος 50 Hz
eeg_clean = 1.5 * np.sin(2 * np.pi * 10 * t) + 2.5 * np.sin(2 * np.pi * 2 * t)
noise_50hz = 1.0 * np.sin(2 * np.pi * 50 * t)
emg_noise = 0.4 * np.random.randn(len(t))
raw_eeg = eeg_clean + noise_50hz + emg_noise

# 2. Εφαρμογή του DSP Pipeline
filtered_notch = apply_notch_filter(raw_eeg, fs=fs, freq=50.0)
filtered_eeg = apply_bandpass_filter(filtered_notch, lowcut=0.5, highcut=35.0, fs=fs)

# 3. Εξαγωγή Band Powers
powers = compute_band_powers(filtered_eeg, fs=fs)

# 4. Οπτικοποίηση
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7))

# Subplot 1: Σήμα στον χρόνο
ax1.plot(t[:400], raw_eeg[:400], label='Raw Noisy EEG (with 50 Hz Hum)', color='gray', alpha=0.6)
ax1.plot(t[:400], filtered_eeg[:400], label='Zero-Phase Filtered EEG (0.5 - 35 Hz)', color='teal', linewidth=1.8)
ax1.set_title('NeuroScan EEG Signal Preprocessing Pipeline', fontsize=12, fontweight='bold')
ax1.set_xlabel('Time (seconds)')
ax1.set_ylabel('Amplitude (µV)')
ax1.legend(loc='upper right')
ax1.grid(True, linestyle='--', alpha=0.5)

# Subplot 2: Κατανομή Ισχύος Εγκεφαλικών Ρυθμών (Bar Chart)
band_names = list(powers.keys())
band_values = list(powers.values())
colors = ['#4c72b0', '#55a868', '#c44e52', '#8172b2']

ax2.bar(band_names, band_values, color=colors, width=0.5)
ax2.set_title('EEG Relative Spectral Band Power (%)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Relative Power (%)')
for i, v in enumerate(band_values):
    ax2.text(i, v + 1.0, f"{v:.1f}%", ha='center', fontweight='bold')
ax2.set_ylim(0, max(band_values) + 15)
ax2.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('assets/eeg_filtering_psd.png', dpi=150)
print("-> DSP Pipeline επιτυχές! Το γράφημα αποθηκεύτηκε στο 'assets/eeg_filtering_psd.png'.")
plt.show()

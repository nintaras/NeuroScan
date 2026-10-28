import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, welch

def apply_notch_filter(signal: np.ndarray, fs: float = 100.0, freq: float = 50.0, q: float = 30.0) -> np.ndarray:
    """Αφαιρεί το θόρυβο των 50 Hz του δικτύου ηλεκτροδότησης."""
    b, a = iirnotch(w0=freq, Q=q, fs=fs)
    return filtfilt(b, a, signal)

def apply_bandpass_filter(signal: np.ndarray, lowcut: float = 0.5, highcut: float = 35.0, fs: float = 100.0, order: int = 4) -> np.ndarray:
    """Εφαρμόζει zero-phase Butterworth bandpass φίλτρο στο φάσμα ενδιαφέροντος του EEG (0.5 - 35 Hz)."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

def compute_band_powers(signal: np.ndarray, fs: float = 100.0) -> dict:
    """
    Υπολογίζει την ισχύ των 4 βασικών εγκεφαλικών ρυθμών (Delta, Theta, Alpha, Beta)
    χρησιμοποιώντας τη μέθοδο Welch (Power Spectral Density).
    """
    freqs, psd = welch(signal, fs=fs, nperseg=int(fs * 2))
    
    bands = {
        'Delta (0.5-4 Hz)': (0.5, 4.0),
        'Theta (4-8 Hz)': (4.0, 8.0),
        'Alpha (8-12 Hz)': (8.0, 12.0),
        'Beta (12-30 Hz)': (12.0, 30.0)
    }
    
    # Συμβατότητα NumPy 2.0+ (np.trapezoid) και παλαιότερων εκδόσεων (np.trapz)
    trapz_fn = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
    
    band_powers = {}
    total_power = trapz_fn(psd, freqs) + 1e-8
    
    for band_name, (low, high) in bands.items():
        idx_band = np.logical_and(freqs >= low, freqs <= high)
        abs_power = trapz_fn(psd[idx_band], freqs[idx_band])
        # Σχετική ισχύς (Relative Band Power % επί του συνόλου)
        band_powers[band_name] = (abs_power / total_power) * 100.0
        
    return band_powers

import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Επιλογή 2: Δυναμική προσθήκη του root directory στο path για ασφαλή imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.dsp import apply_bandpass_filter, apply_notch_filter

class SleepEEGDataset(Dataset):
    """
    Dataset για πολυκαναλικό/μονοκαναλικό EEG τεμαχισμένο σε 30-second epochs (3000 samples @ 100Hz).
    Στάδια Ύπνου:
      0: Wake (Alpha/Beta dominance)
      1: Light/REM Sleep (Theta dominance + Spindles)
      2: Deep Sleep / SWS (Delta dominance, high amplitude)
    """
    def __init__(self, num_epochs: int = 600, fs: float = 100.0, seed: int = 42):
        np.random.seed(seed)
        self.fs = fs
        self.num_epochs = num_epochs
        self.epoch_len = int(fs * 30)  # 3000 samples per epoch
        
        self.signals = []
        self.labels = []
        self._generate_eeg_data()

    def _generate_eeg_data(self):
        t = np.linspace(0, 30, self.epoch_len, endpoint=False)
        
        for _ in range(self.num_epochs):
            # Ισορροπημένη κατανομή σταδίων (Wake, Light, Deep)
            stage = np.random.choice([0, 1, 2])
            
            if stage == 0:  # Wake
                # Alpha (10 Hz) + Beta (20 Hz)
                clean = 1.2 * np.sin(2 * np.pi * 10 * t) + 0.6 * np.sin(2 * np.pi * 20 * t)
            elif stage == 1:  # Light / REM
                # Theta (6 Hz) + σύντομα Sleep Spindles (13 Hz)
                spindle = np.sin(2 * np.pi * 13 * t) * np.exp(-((t - 15) ** 2) / 2)
                clean = 1.0 * np.sin(2 * np.pi * 6 * t) + 1.5 * spindle
            else:  # Deep Sleep (Delta dominance)
                # Υψηλού πλάτους Delta (1.5 Hz)
                clean = 3.5 * np.sin(2 * np.pi * 1.5 * t) + 0.3 * np.random.randn(self.epoch_len)
            
            # Προσθήκη φυσιολογικού θορύβου και 50 Hz powerline hum
            noise = 0.5 * np.sin(2 * np.pi * 50 * t) + 0.4 * np.random.randn(self.epoch_len)
            raw = clean + noise
            
            # Εφαρμογή DSP Pipeline (Notch + Zero-phase Bandpass 0.5-35 Hz)
            filtered = apply_notch_filter(raw, fs=self.fs, freq=50.0)
            filtered = apply_bandpass_filter(filtered, lowcut=0.5, highcut=35.0, fs=self.fs)
            
            # Z-Score Normalization ανά epoch
            norm_signal = (filtered - np.mean(filtered)) / (np.std(filtered) + 1e-8)
            
            self.signals.append(norm_signal)
            self.labels.append(stage)

    def __len__(self):
        return self.num_epochs

    def __getitem__(self, idx):
        # Tensor Shape: (1, 3000) -> 1 Channel, 3000 time steps
        signal_tensor = torch.tensor(self.signals[idx], dtype=torch.float32).unsqueeze(0)
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)
        return signal_tensor, label_tensor

if __name__ == "__main__":
    dataset = SleepEEGDataset(num_epochs=100)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    batch_signals, batch_labels = next(iter(loader))
    print(f"Batch Signals Tensor Shape: {batch_signals.shape}")
    print(f"Batch Labels Tensor Shape: {batch_labels.shape}")
    print(f"Labels Sample: {batch_labels[:8].tolist()}")
    print("-> PyTorch SleepEEGDataset επαληθεύτηκε επιτυχώς!")

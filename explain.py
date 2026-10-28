import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

from src.dataset import SleepEEGDataset
from src.model import SleepStageCRNN

class EEGGradCAM:
    """1D Grad-CAM για την εξαγωγή activation heatmaps από το τελευταίο Conv layer του CRNN."""
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._hook_layers()

    def _hook_layers(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, target_class=None):
        self.model.eval()
        self.model.zero_grad()

        output = self.model(input_tensor)
        if target_class is None:
            target_class = torch.argmax(output, dim=1).item()

        loss = output[0, target_class]
        loss.backward()

        # Global average pooling στα gradients
        weights = torch.mean(self.gradients, dim=2, keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)

        # Interpolation στο αρχικό μήκος των 3000 δειγμάτων
        cam = F.interpolate(cam, size=input_tensor.shape[2], mode='linear', align_corners=False)
        cam = cam.squeeze().cpu().detach().numpy()
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-8)
        return cam, target_class

def run_explainability():
    device = torch.device("cpu")  # CPU για ασφαλή backward hooks
    
    # 1. Φόρτωση εκπαιδευμένου μοντέλου
    model = SleepStageCRNN(in_channels=1, num_classes=3, lstm_hidden=64)
    model.load_state_dict(torch.load("best_neuroscan_model.pth", map_location=device))
    model.eval()

    # 2. Φόρτωση ενός δείγματος βαθιού ύπνου (Deep SWS - Class 2) ή Light Sleep (Class 1)
    dataset = SleepEEGDataset(num_epochs=50, fs=100.0, seed=10)
    # Επιλέγουμε ένα δείγμα Deep Sleep (Label 2) για καθαρή απεικόνιση κυμάτων Delta
    sample_idx = 0
    for idx in range(len(dataset)):
        sig, lbl = dataset[idx]
        if lbl.item() == 2:
            sample_idx = idx
            break

    signal_tensor, label = dataset[sample_idx]
    signal_tensor = signal_tensor.unsqueeze(0).to(device)  # (1, 1, 3000)

    # 3. Εκτέλεση 1D Grad-CAM πάνω στο 3ο Conv layer (features[8])
    target_layer = model.features[8]  # Το τελευταίο Conv1d layer πριν το MaxPool
    grad_cam = EEGGradCAM(model, target_layer)
    heatmap, pred_class = grad_cam.generate(signal_tensor)

    class_names = ["Wake", "Light/REM", "Deep Sleep (SWS)"]
    raw_signal = signal_tensor.squeeze().numpy()
    fs = 100.0
    time_axis = np.linspace(0, 30, len(raw_signal))

    # 4. Υπολογισμός Spectrogram (STFT)
    f, t_spec, Sxx = spectrogram(raw_signal, fs=fs, nperseg=128, noverlap=64)
    freq_mask = f <= 30.0  # Περιορισμός στις συχνότητες ενδιαφέροντος 0-30 Hz

    # 5. Οπτικοποίηση 3 Επιπέδων
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 9), sharex=False)

    # Επίπεδο 1: Σήμα στον χρόνο
    ax1.plot(time_axis, raw_signal, color='#1f77b4', linewidth=1.2)
    ax1.set_title(f"NeuroScan: 30s EEG Epoch | True Label: {class_names[label]} | Prediction: {class_names[pred_class]}", 
                  fontsize=12, fontweight='bold')
    ax1.set_ylabel("Normalized Amp")
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Επίπεδο 2: Time-Frequency Spectrogram
    im = ax2.pcolormesh(t_spec, f[freq_mask], 10 * np.log10(Sxx[freq_mask, :] + 1e-10), 
                        shading='gouraud', cmap='viridis')
    ax2.set_title("Short-Time Fourier Transform (STFT) Spectrogram", fontsize=11, fontweight='bold')
    ax2.set_ylabel("Frequency (Hz)")
    fig.colorbar(im, ax=ax2, label="Power (dB)")

    # Επίπεδο 3: Grad-CAM Saliency Map
    ax3.plot(time_axis, raw_signal, color='gray', alpha=0.4, label='EEG Waveform')
    scatter = ax3.scatter(time_axis, raw_signal, c=heatmap, cmap='plasma', s=6, label='Grad-CAM Attention')
    ax3.set_title("1D Grad-CAM Explainability: Salient Morphological Attributions", fontsize=11, fontweight='bold')
    ax3.set_xlabel("Time (seconds)")
    ax3.set_ylabel("Amplitude")
    fig.colorbar(scatter, ax=ax3, label="Relevance Weight")
    ax3.legend(loc='upper right')
    ax3.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig("assets/explainability_gradcam.png", dpi=150)
    print("-> Το γράφημα explainability αποθηκεύτηκε στο 'assets/explainability_gradcam.png'.")
    plt.show()

if __name__ == "__main__":
    run_explainability()

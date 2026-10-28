import torch
import torch.nn as nn

class SleepStageCRNN(nn.Module):
    """
    Υβριδική αρχιτεκτονική 1D-CNN + BiLSTM για ταξινόμηση σημάτων EEG σε στάδια ύπνου.
    Input Shape:  (Batch, 1, 3000)
    Output Shape: (Batch, num_classes)
    """
    def __init__(self, in_channels: int = 1, num_classes: int = 3, lstm_hidden: int = 64):
        super(SleepStageCRNN, self).__init__()
        
        # 1D-CNN Feature Extractor
        self.features = nn.Sequential(
            # Block 1: Μεγάλος πυρήνας (kernel_size=15) για καταγραφή χαμηλών συχνοτήτων (Delta)
            nn.Conv1d(in_channels, 16, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),  # 3000 -> 1500 -> 375
            
            # Block 2: Μεσαίος πυρήνας (kernel_size=7) για κύματα Theta και Alpha
            nn.Conv1d(16, 32, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),  # 375 -> 93
            
            # Block 3: Μικρός πυρήνας (kernel_size=3) για γρήγορες μεταβάσεις (Spindles/Beta)
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)   # 93 -> 46
        )
        
        # BiLSTM Sequence Modeling
        # Είσοδος στο LSTM: (Batch, Sequence Length, Features) -> (Batch, 46, 64)
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        # Classifier Head (bidirectional -> hidden_size * 2)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. CNN Feature Extraction: (B, 1, 3000) -> (B, 64, 46)
        feat = self.features(x)
        
        # 2. Reshape για LSTM: (B, Channels, Length) -> (B, Length, Channels)
        feat = feat.permute(0, 2, 1)
        
        # 3. BiLSTM Forward Pass
        lstm_out, _ = self.lstm(feat)  # (B, 46, 128)
        
        # 4. Global Temporal Pooling (μέσος όρος πάνω στα χρονικά βήματα)
        pooled = torch.mean(lstm_out, dim=1)  # (B, 128)
        
        # 5. Τελική ταξινόμηση
        logits = self.classifier(pooled)      # (B, 3)
        return logits

if __name__ == "__main__":
    # Smoke test ελέγχου διαστάσεων
    model = SleepStageCRNN(in_channels=1, num_classes=3)
    dummy_input = torch.randn(8, 1, 3000)  # Batch=8, 1 Channel, 3000 Samples
    output = model(dummy_input)
    
    print("-> Model Architecture Loaded Successfully!")
    print(f"Input Shape:  {dummy_input.shape}")
    print(f"Output Shape: {output.shape} (Batch, Num Classes)")

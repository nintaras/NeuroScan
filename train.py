import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

from src.dataset import SleepEEGDataset
from src.model import SleepStageCRNN

def train_neuroscan():
    # 1. Hardware acceleration (MPS για Apple Silicon Mac, αλλιώς CPU)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"-> Εκπαίδευση σε συσκευή: {device}")

    # 2. Φόρτωση και διαχωρισμός Dataset
    full_dataset = SleepEEGDataset(num_epochs=600, fs=100.0, seed=42)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_set, val_set = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=16, shuffle=False)

    # 3. Αρχικοποίηση Μοντέλου, Loss και Optimizer
    model = SleepStageCRNN(in_channels=1, num_classes=3, lstm_hidden=64).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # 4. Training Loop
    epochs = 10
    best_val_acc = 0.0
    os.makedirs("assets", exist_ok=True)

    print("\n--- Έναρξη Εκπαίδευσης SleepStageCRNN ---")
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for signals, labels in train_loader:
            signals, labels = signals.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(signals)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * signals.size(0)
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_loss = running_loss / total_train
        train_acc = (correct_train / total_train) * 100.0

        # Validation Phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for signals, labels in val_loader:
                signals, labels = signals.to(device), labels.to(device)
                outputs = model(signals)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * signals.size(0)
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        val_loss = val_loss / total_val
        val_acc = (correct_val / total_val) * 100.0

        print(f"Epoch [{epoch:02d}/{epochs:02d}] | "
              f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.1f}% | "
              f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.1f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_neuroscan_model.pth")

    print(f"\n-> Η εκπαίδευση ολοκληρώθηκε! Καλύτερο Val Acc: {best_val_acc:.1f}%")
    print("-> Το μοντέλο αποθηκεύτηκε ως 'best_neuroscan_model.pth'.")

    # 5. Τελική Αξιολόγηση & Confusion Matrix
    model.load_state_dict(torch.load("best_neuroscan_model.pth"))
    model.eval()

    all_preds = []
    all_targets = []
    with torch.no_grad():
        for signals, labels in val_loader:
            signals = signals.to(device)
            outputs = model(signals)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.numpy())

    class_names = ["Wake", "Light/REM", "Deep (SWS)"]
    print("\n--- Classification Report ---")
    print(classification_report(all_targets, all_preds, target_names=class_names))

    # Οπτικοποίηση και Αποθήκευση Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap="Blues", ax=ax, values_format="d")
    ax.set_title("NeuroScan: Sleep Stage Confusion Matrix", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig("assets/confusion_matrix.png", dpi=150)
    print("-> Ο πίνακας σύγχυσης αποθηκεύτηκε στο 'assets/confusion_matrix.png'.")
    plt.show()

if __name__ == "__main__":
    train_neuroscan()

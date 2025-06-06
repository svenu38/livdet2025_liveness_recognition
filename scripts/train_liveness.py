import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import pandas as pd
from tqdm import tqdm

# Add root to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.liveness_model import LivenessModel
from scripts.utils import FingerprintDataset

# === CONFIG ===
data_csv = "data/processed/train.csv"
val_csv = "data/processed/val.csv"
img_size = 256
batch_size = 32
epochs = 20
lr = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === TRANSFORMS ===
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor()
])

# === LOAD DATA ===
train_df = pd.read_csv(data_csv)
val_df = pd.read_csv(val_csv)

train_dataset = FingerprintDataset(train_df, transform)
val_dataset = FingerprintDataset(val_df, transform)

# 🔧 Set num_workers=0 for macOS multiprocessing compatibility
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# === TRAINING SCRIPT ===
if __name__ == "__main__":
    model = LivenessModel().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_val_acc = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            imgs, labels = imgs.to(device), labels.float().to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # === VALIDATION ===
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                outputs = model(imgs)
                val_preds.extend((outputs > 0.5).int().cpu().numpy())
                val_targets.extend(labels.numpy())

        val_acc = accuracy_score(val_targets, val_preds)
        print(f"\nEpoch {epoch+1} - Loss: {running_loss/len(train_loader):.4f} - Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "models/best_liveness_model.pth")
            print("✅ Best model saved!")

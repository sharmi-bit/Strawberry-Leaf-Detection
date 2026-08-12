"""
Trains the U-Net segmentation model on the (synthetic, by default) dataset.

Usage:
    python train_unet.py                       # uses backend/data/synthetic_dataset.npz
    python train_unet.py --data path/to.npz     # or your own real dataset
    python train_unet.py --epochs 30

Expected .npz contents (see generate_synthetic_data.py for the reference
implementation):
    images: (N, H, W, 3) uint8 RGB
    masks:  (N, H, W)    uint8 {0,1} lesion mask
    labels: (N,)         int64 class id (used only by train_xgboost.py)
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from models.unet import UNet  # noqa: E402


class LeafSegDataset(Dataset):
    def __init__(self, images, masks):
        self.images = images
        self.masks = masks

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx].astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)  # CHW
        mask = torch.from_numpy(self.masks[idx].astype(np.float32)).unsqueeze(0)
        return img, mask


def dice_loss(logits, target, eps=1e-6):
    probs = torch.sigmoid(logits)
    probs = probs.view(probs.size(0), -1)
    target = target.view(target.size(0), -1)
    intersection = (probs * target).sum(dim=1)
    union = probs.sum(dim=1) + target.sum(dim=1)
    dice = (2 * intersection + eps) / (union + eps)
    return 1 - dice.mean()


def train(data_path, out_path, epochs=25, batch_size=16, lr=1e-3, device="cpu"):
    data = np.load(data_path)
    images, masks = data["images"], data["masks"]

    dataset = LeafSegDataset(images, masks)
    n_val = max(1, int(0.15 * len(dataset)))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = UNet(in_channels=3, out_channels=1, base_ch=16).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()

    best_val = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for imgs, msks in train_loader:
            imgs, msks = imgs.to(device), msks.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = bce(logits, msks) + dice_loss(logits, msks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
        train_loss /= n_train

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, msks in val_loader:
                imgs, msks = imgs.to(device), msks.to(device)
                logits = model(imgs)
                loss = bce(logits, msks) + dice_loss(logits, msks)
                val_loss += loss.item() * imgs.size(0)
        val_loss /= n_val

        print(f"epoch {epoch:02d}/{epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            torch.save(model.state_dict(), out_path)

    print(f"Best val_loss={best_val:.4f} -> saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=os.path.join(os.path.dirname(__file__), "..", "data", "synthetic_dataset.npz"))
    parser.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "weights", "unet_leaf_segmentation.pth"))
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on device: {device}")
    train(args.data, args.out, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, device=device)

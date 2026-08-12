"""
Trains the XGBoost disease classifier on features extracted from the U-Net
lesion masks.

Pipeline:
  1. Load dataset (images + ground-truth masks + class labels)
  2. Run the trained U-Net to predict a lesion mask for every image
     (falls back to the ground-truth mask if no U-Net checkpoint is found,
     which is useful for a quick sanity-check run)
  3. Extract the feature vector for every image (models/feature_extraction.py)
  4. Train a multi-class XGBoost classifier on (features -> class label)
  5. Save the model + a label map

Usage:
    python train_xgboost.py
    python train_xgboost.py --data path/to.npz --unet path/to/unet.pth
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import xgboost as xgb
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from models.unet import UNet  # noqa: E402
from models.feature_extraction import extract_features, FEATURE_NAMES  # noqa: E402


def predict_masks(images, unet_path, device="cpu", threshold=0.5):
    model = UNet(in_channels=3, out_channels=1, base_ch=16).to(device)
    model.load_state_dict(torch.load(unet_path, map_location=device))
    model.eval()

    masks = []
    with torch.no_grad():
        for i in range(0, len(images), 32):
            batch = images[i:i + 32].astype(np.float32) / 255.0
            batch_t = torch.from_numpy(batch).permute(0, 3, 1, 2).to(device)
            logits = model(batch_t)
            probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
            masks.append((probs > threshold).astype(np.uint8))
    return np.concatenate(masks, axis=0)


def build_feature_matrix(images, masks):
    feats = np.stack([extract_features(img, m) for img, m in zip(images, masks)])
    return feats


def main(data_path, unet_path, out_dir):
    data = np.load(data_path, allow_pickle=True)
    images, gt_masks, labels = data["images"], data["masks"], data["labels"]

    if "classes" in data:
        classes = [str(c) for c in data["classes"]]
    else:
        # fall back to the synthetic dataset's class order
        from train.generate_synthetic_data import CLASSES as classes  # noqa: E402

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if os.path.exists(unet_path):
        print(f"Using U-Net predicted masks from {unet_path}")
        masks = predict_masks(images, unet_path, device=device)
    else:
        print("No U-Net checkpoint found - falling back to ground-truth masks for feature extraction")
        masks = gt_masks

    print("Extracting features...")
    X = build_feature_matrix(images, masks)
    y = labels

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="mlogloss" if len(classes) > 2 else "logloss",
        random_state=42,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=classes, digits=3)
    print(report)

    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, "xgboost_classifier.json")
    clf.save_model(model_path)

    meta = {
        "classes": classes,
        "feature_names": FEATURE_NAMES,
        "unet_base_ch": 16,
    }
    with open(os.path.join(out_dir, "model_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    with open(os.path.join(out_dir, "classification_report.txt"), "w") as f:
        f.write(report)

    print(f"Saved XGBoost model -> {model_path}")
    print(f"Saved metadata -> {os.path.join(out_dir, 'model_meta.json')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    base = os.path.dirname(__file__)
    parser.add_argument("--data", default=os.path.join(base, "..", "data", "synthetic_dataset.npz"))
    parser.add_argument("--unet", default=os.path.join(base, "..", "weights", "unet_leaf_segmentation.pth"))
    parser.add_argument("--out", default=os.path.join(base, "..", "weights"))
    args = parser.parse_args()
    main(args.data, args.unet, args.out)

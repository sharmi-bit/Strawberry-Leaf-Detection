"""
Generates a synthetic strawberry-leaf dataset (images + lesion masks + class
labels) used ONLY to make this repo runnable end-to-end out of the box.

IMPORTANT: This is a procedural stand-in, not real leaf photography. It exists
so `train_unet.py` and `train_xgboost.py` produce real, loadable weight files
and the FastAPI + frontend demo can be exercised immediately. For a model
that is actually accurate on real strawberry leaves, replace this step by
pointing the training scripts at your real dataset (see README "Training on
your real dataset").

Classes:
  0 = healthy
  1 = angular_leaf_spot   (bacterial - angular, dark polygonal lesions)
  2 = leaf_scorch         (fungal - purple/red scorched blotches near edges)
  3 = leaf_spot           (fungal - round dark-purple spots with defined rings)
"""

import os
import random

import cv2
import numpy as np

CLASSES = ["healthy", "angular_leaf_spot", "leaf_scorch", "leaf_spot"]
IMG_SIZE = 96


def _leaf_base(rng):
    """Base leaf-green canvas with a soft radial vignette + vein texture."""
    img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
    base_g = rng.uniform(90, 150)
    img[..., 0] = rng.uniform(20, 50)   # R
    img[..., 1] = base_g                 # G
    img[..., 2] = rng.uniform(20, 45)   # B

    # subtle mottling
    noise = rng.normal(0, 8, (IMG_SIZE, IMG_SIZE, 1)).astype(np.float32)
    img += noise

    # simple central vein
    cv2.line(img, (IMG_SIZE // 2, 5), (IMG_SIZE // 2, IMG_SIZE - 5),
              (base_g * 0.6, base_g * 0.8, base_g * 0.5), 2)
    return img


def _add_angular_spots(img, mask, rng, n=None):
    n = n or rng.integers(3, 7)
    for _ in range(n):
        cx, cy = rng.integers(15, IMG_SIZE - 15, size=2)
        r = rng.integers(4, 10)
        pts = []
        sides = rng.integers(5, 7)
        for i in range(sides):
            ang = 2 * np.pi * i / sides + rng.uniform(-0.3, 0.3)
            rr = r * rng.uniform(0.7, 1.1)
            pts.append((int(cx + rr * np.cos(ang)), int(cy + rr * np.sin(ang))))
        pts = np.array([pts], dtype=np.int32)
        color = (rng.uniform(30, 55), rng.uniform(25, 45), rng.uniform(60, 90))
        cv2.fillPoly(img, pts, color)
        cv2.fillPoly(mask, pts, 1)
    return img, mask


def _add_scorch(img, mask, rng, n=None):
    n = n or rng.integers(2, 4)
    for _ in range(n):
        # blotches concentrated near the border
        edge = rng.choice(["top", "bottom", "left", "right"])
        if edge == "top":
            cx, cy = rng.integers(10, IMG_SIZE - 10), rng.integers(3, 15)
        elif edge == "bottom":
            cx, cy = rng.integers(10, IMG_SIZE - 10), rng.integers(IMG_SIZE - 15, IMG_SIZE - 3)
        elif edge == "left":
            cx, cy = rng.integers(3, 15), rng.integers(10, IMG_SIZE - 10)
        else:
            cx, cy = rng.integers(IMG_SIZE - 15, IMG_SIZE - 3), rng.integers(10, IMG_SIZE - 10)
        axes = (rng.integers(8, 18), rng.integers(6, 14))
        angle = rng.uniform(0, 180)
        color = (rng.uniform(45, 70), rng.uniform(15, 35), rng.uniform(70, 100))
        cv2.ellipse(img, (cx, cy), axes, angle, 0, 360, color, -1)
        cv2.ellipse(mask, (cx, cy), axes, angle, 0, 360, 1, -1)
    return img, mask


def _add_round_spots(img, mask, rng, n=None):
    n = n or rng.integers(4, 9)
    for _ in range(n):
        cx, cy = rng.integers(10, IMG_SIZE - 10, size=2)
        r = rng.integers(3, 8)
        # dark purple center
        color_center = (rng.uniform(35, 55), rng.uniform(20, 35), rng.uniform(65, 95))
        cv2.circle(img, (cx, cy), r, color_center, -1)
        cv2.circle(mask, (cx, cy), r, 1, -1)
        # lighter ring (still counted as lesion)
        ring_color = (rng.uniform(70, 100), rng.uniform(80, 110), rng.uniform(50, 70))
        cv2.circle(img, (cx, cy), r + 2, ring_color, 1)
    return img, mask


def generate_sample(rng, label_idx):
    img = _leaf_base(rng)
    mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)

    if label_idx == 0:  # healthy
        pass
    elif label_idx == 1:  # angular leaf spot
        img, mask = _add_angular_spots(img, mask, rng)
    elif label_idx == 2:  # leaf scorch
        img, mask = _add_scorch(img, mask, rng)
    elif label_idx == 3:  # leaf spot
        img, mask = _add_round_spots(img, mask, rng)

    img = np.clip(img, 0, 255).astype(np.uint8)
    return img, mask


def build_dataset(n_per_class=200, seed=42, out_dir=None):
    rng = np.random.default_rng(seed)
    images, masks, labels = [], [], []
    for label_idx in range(len(CLASSES)):
        for _ in range(n_per_class):
            img, mask = generate_sample(rng, label_idx)
            images.append(img)
            masks.append(mask)
            labels.append(label_idx)

    images = np.stack(images)
    masks = np.stack(masks)
    labels = np.array(labels, dtype=np.int64)

    # shuffle
    perm = rng.permutation(len(labels))
    images, masks, labels = images[perm], masks[perm], labels[perm]

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        np.savez_compressed(
            os.path.join(out_dir, "synthetic_dataset.npz"),
            images=images, masks=masks, labels=labels,
        )
    return images, masks, labels


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(__file__), "..", "data")
    imgs, msks, lbls = build_dataset(n_per_class=200, out_dir=out)
    print(f"Generated {len(lbls)} samples -> {out}/synthetic_dataset.npz")
    for i, c in enumerate(CLASSES):
        print(f"  {c}: {(lbls == i).sum()}")

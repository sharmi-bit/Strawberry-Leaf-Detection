"""
Builds the final REAL multi-class dataset by combining:

  1. The Kaggle "Strawberry Disease Detection Dataset" (Afzaal et al., 2021,
     Sensors) - LabelMe-style per-image JSON polygon annotations. We use
     only the three LEAF-specific disease classes it contains:
       - Angular Leafspot
       - Leaf Spot
       - Powdery Mildew Leaf
     (Gray Mold, Blossom Blight, Powdery Mildew Fruit, Anthracnose Fruit Rot
     are fruit/flower diseases and out of scope for a leaf-disease project.)
     Polygons are rasterized into real, ground-truth lesion masks - no
     pseudo-labelling needed for this part.

  2. The already-prepared PlantVillage real dataset (healthy + leaf_scorch),
     built earlier by prepare_real_data.py.

Only images with a SINGLE disease label are used (a handful of images in
the Kaggle set have more than one disease type annotated - those are
skipped to keep every image single-labelled, matching how the rest of the
pipeline works).

Usage:
    python prepare_kaggle_data.py --kaggle_dir /path/to/train --plantvillage_npz ../data/real_strawberry_dataset.npz --out ../data/combined_real_dataset.npz
"""

import argparse
import glob
import json
import os

import cv2
import numpy as np

IMG_SIZE = 96

# unified 5-class scheme across both real data sources
CLASSES = ["healthy", "leaf_scorch", "angular_leafspot", "leaf_spot", "powdery_mildew_leaf"]

KAGGLE_LABEL_TO_CLASS = {
    "Angular Leafspot": "angular_leafspot",
    "Leaf Spot": "leaf_spot",
    "Powdery Mildew Leaf": "powdery_mildew_leaf",
}


def load_kaggle_split(kaggle_dir, max_per_class=None):
    json_files = glob.glob(os.path.join(kaggle_dir, "*.json"))
    per_class_count = {c: 0 for c in KAGGLE_LABEL_TO_CLASS.values()}

    images, masks, labels = [], [], []
    skipped_mixed = 0
    skipped_other = 0

    for jf in sorted(json_files):
        with open(jf) as f:
            d = json.load(f)

        shape_labels = set(s["label"] for s in d["shapes"])
        if len(shape_labels) != 1:
            skipped_mixed += 1
            continue

        raw_label = list(shape_labels)[0]
        if raw_label not in KAGGLE_LABEL_TO_CLASS:
            skipped_other += 1
            continue

        class_name = KAGGLE_LABEL_TO_CLASS[raw_label]
        if max_per_class and per_class_count[class_name] >= max_per_class:
            continue

        img_path = os.path.join(kaggle_dir, d["imagePath"])
        if not os.path.exists(img_path):
            # imagePath may have different casing/extension in some rows
            candidate = os.path.splitext(jf)[0] + ".jpg"
            if os.path.exists(candidate):
                img_path = candidate
            else:
                continue

        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = d["imageHeight"], d["imageWidth"]

        # rasterize all polygons for this image into one binary lesion mask
        mask = np.zeros((h, w), dtype=np.uint8)
        for shape in d["shapes"]:
            pts = np.array(shape["points"], dtype=np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(mask, [pts], 1)

        img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        mask_resized = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)

        rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        images.append(rgb)
        masks.append(mask_resized)
        labels.append(CLASSES.index(class_name))
        per_class_count[class_name] += 1

    print(f"Kaggle: loaded {len(images)} images, skipped {skipped_mixed} mixed-label, {skipped_other} out-of-scope")
    for c, n in per_class_count.items():
        print(f"  {c}: {n}")

    return images, masks, labels


def main(kaggle_dir, plantvillage_npz, out_path, max_per_class):
    k_images, k_masks, k_labels = load_kaggle_split(kaggle_dir, max_per_class)

    pv_data = np.load(plantvillage_npz, allow_pickle=True)
    pv_images, pv_masks, pv_labels_old = pv_data["images"], pv_data["masks"], pv_data["labels"]
    pv_classes_old = [str(c) for c in pv_data["classes"]]  # ["healthy", "leaf_scorch"]

    # remap PlantVillage's old label ids (0/1) onto the unified CLASSES order
    remap = {i: CLASSES.index(name) for i, name in enumerate(pv_classes_old)}
    pv_labels = np.array([remap[l] for l in pv_labels_old.tolist()], dtype=np.int64)

    all_images = list(pv_images) + k_images
    all_masks = list(pv_masks) + k_masks
    all_labels = pv_labels.tolist() + k_labels

    images = np.stack(all_images).astype(np.uint8)
    masks = np.stack(all_masks).astype(np.uint8)
    labels = np.array(all_labels, dtype=np.int64)

    rng = np.random.default_rng(42)
    perm = rng.permutation(len(labels))
    images, masks, labels = images[perm], masks[perm], labels[perm]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, images=images, masks=masks, labels=labels, classes=np.array(CLASSES))

    print(f"\nSaved combined dataset -> {out_path}")
    print(f"Total: {len(labels)} images across {len(CLASSES)} classes")
    for i, c in enumerate(CLASSES):
        print(f"  {c}: {(labels == i).sum()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    base = os.path.dirname(__file__)
    parser.add_argument("--kaggle_dir", required=True, help="Path to the Kaggle dataset's train/ folder")
    parser.add_argument("--plantvillage_npz", default=os.path.join(base, "..", "data", "real_strawberry_dataset.npz"))
    parser.add_argument("--out", default=os.path.join(base, "..", "data", "combined_real_dataset.npz"))
    parser.add_argument("--max_per_class", type=int, default=None, help="Cap Kaggle images per class (for speed)")
    args = parser.parse_args()
    main(args.kaggle_dir, args.plantvillage_npz, args.out, args.max_per_class)

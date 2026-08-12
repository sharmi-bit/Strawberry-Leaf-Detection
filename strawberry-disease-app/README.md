# Strawberry Leaf Disease Detection &mdash; U-Net + XGBoost

A two-stage pipeline for strawberry leaf disease detection, wired up to a working FastAPI backend and a frontend you can open in a browser:

1. **U-Net** (PyTorch) segments the leaf image to find diseased/lesion pixels.
2. **Feature extraction** turns that segmentation into a numeric feature vector: lesion geometry (area ratio, region count, perimeter/area), colour statistics (RGB + HSV, whole-leaf and lesion-only), and GLCM texture (contrast, homogeneity, energy, correlation).
3. **XGBoost** classifies the leaf using those features, across 5 real classes: `healthy`, `leaf_scorch`, `angular_leafspot`, `leaf_spot`, `powdery_mildew_leaf`. The code supports any number of classes if you retrain on a different/larger dataset.

```
strawberry-disease-app/
├── backend/
│   ├── app.py                       # FastAPI server (/health, /predict)
│   ├── requirements.txt
│   ├── models/
│   │   ├── unet.py                  # U-Net architecture
│   │   └── feature_extraction.py    # mask -> feature vector
│   ├── train/
│   │   ├── generate_synthetic_data.py
│   │   ├── prepare_real_data.py       # PlantVillage -> healthy/leaf_scorch + pseudo-masks
│   │   ├── prepare_kaggle_data.py     # Kaggle (Afzaal et al.) -> 3 leaf disease classes + real masks
│   │   ├── train_unet.py
│   │   └── train_xgboost.py
│   ├── data/
│   │   ├── real_strawberry_dataset.npz     # PlantVillage: healthy + leaf_scorch
│   │   └── combined_real_dataset.npz       # merged 5-class dataset used for current weights
│   └── weights/                       # trained model files (already included, real-data trained)
└── frontend/
    └── index.html                   # single-file UI, talks to the backend API
```

## ⚠️ Important — read this first

The zip you originally uploaded (`Strawberry_Leaf_Disease_Detection-main`) only contained CNN/ResNet34 training notebooks with **no dataset and no saved weights** — it didn't match the U-Net + XGBoost pipeline from your paper. So this project was rebuilt from scratch with the correct architecture (real U-Net, real feature-engineering, real XGBoost).

**Current weights are trained on 1,399 real photos across 5 real classes**, combined from two sources:

| Source | Classes | Images | Masks |
|---|---|---|---|
| PlantVillage (GitHub mirror) | `healthy`, `leaf_scorch` | 799 | Pseudo-masks (colour-threshold, see `prepare_real_data.py`) |
| Kaggle — Afzaal et al. 2021 (`usmanafzaal/strawberry-disease-detection-dataset`) | `angular_leafspot`, `leaf_spot`, `powdery_mildew_leaf` | 600 | **Real ground-truth polygon masks**, rasterized from the dataset's LabelMe annotations |

**Test accuracy: 99.3% across all 5 classes** (280 held-out real images — see `backend/weights/classification_report.txt`). Spot-checked with 9 additional real photos the model had never seen at all (not in train or test split): 9/9 correct, all above 92% confidence.

The Kaggle dataset actually has **7** disease classes total, but only these 3 are leaf diseases — the rest (`gray_mold`, `blossom_blight`, `powdery_mildew_fruit`, `anthracnose_fruit_rot`) affect fruit or flowers, out of scope for a leaf-disease project, and were left out. It also has no "healthy" class of its own (it's disease-instance segmentation, not healthy-vs-sick classification), which is why the PlantVillage `healthy` class is what anchors that end of the classifier.

**A note on the pseudo-masks:** the PlantVillage portion (`healthy`/`leaf_scorch`) still uses masks derived from a colour heuristic rather than hand-drawn ground truth, since PlantVillage doesn't ship pixel-level annotations. The 3 Kaggle-derived classes use real, hand-drawn segmentation polygons. If you want fully ground-truth-masked training data end to end, either source a masked healthy/scorch dataset, or drop `leaf_scorch` and rely only on the 3 Kaggle classes plus a "healthy" class you curate yourself.

**A note on the leaf-vs-not-a-leaf guard:** `/predict` now checks whether the uploaded image is plausibly a leaf at all (a cheap green-hue-fraction heuristic in `app.py`'s `looks_like_leaf()`) before running the full pipeline, and returns a 422 error instead of forcing a diagnosis if it isn't. It's a heuristic, not a trained classifier, so it's not bulletproof — but it stops the common case of someone accidentally uploading a photo of a person or an unrelated object and getting a confident wrong diagnosis back.

## Retraining further

If you get the richer 7-class Kaggle dataset (or any dataset with pixel-level lesion masks):

1. Get it into this shape and save as an `.npz`:
   - `images`: `(N, H, W, 3)` uint8 RGB, all leaves resized to the same size (96×96 was used here)
   - `masks`: `(N, H, W)` uint8, `1` = lesion/diseased pixel, `0` = healthy pixel
   - `labels`: `(N,)` int64 class id
   - `classes`: array of class-name strings, in label-id order (e.g. `np.array(["healthy", "angular_leaf_spot", "leaf_scorch", "leaf_spot"])`) — the training scripts and API read this directly, so any number of classes works with no code changes

   ```python
   import numpy as np
   np.savez_compressed("real_dataset.npz", images=images, masks=masks, labels=labels, classes=classes)
   ```

2. Train both models on it:
   ```bash
   cd backend
   python train/train_unet.py --data path/to/real_dataset.npz --epochs 40
   python train/train_xgboost.py --data path/to/real_dataset.npz
   ```
   This overwrites `weights/unet_leaf_segmentation.pth`, `weights/xgboost_classifier.json`, and `weights/model_meta.json` in place — the API and frontend automatically pick up the new weights (and new class list) on restart. If your dataset doesn't have masks but does have PlantVillage-style paired background-removed images, `backend/train/prepare_real_data.py` shows the pseudo-mask approach used for the current weights and can be adapted.

If you only have image-level disease labels (no masks at all, no background-removed pairs), the honest fallback is to drop U-Net and go straight from image → CNN features → XGBoost. Happy to build that variant if that's closer to what your dataset looks like.

## Running it locally

```bash
cd backend
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```

Then open `frontend/index.html` directly in a browser (double-click it, or serve it with any static server). The API URL field at the bottom of the upload panel defaults to `http://localhost:8000` — change it if you deploy the backend elsewhere.

### API

- `GET /health` → `{ status, models_loaded, device }`
- `POST /predict` → multipart form with a `file` field (jpg/png/webp). Returns:
  ```json
  {
    "predicted_class": "leaf_spot",
    "confidence": 0.94,
    "class_probabilities": { "healthy": 0.01, "angular_leaf_spot": 0.02, "leaf_scorch": 0.03, "leaf_spot": 0.94 },
    "lesion_area_percent": 5.3,
    "overlay_image_base64": "..."
  }
  ```

## Deploying

- **Backend**: any host that runs a FastAPI/uvicorn app (Render, Railway, Fly.io, a small EC2/VM, etc). CPU-only inference is fine — this is a small U-Net.
- **Frontend**: `frontend/index.html` is a static file with no build step, so it can go on GitHub Pages, Netlify, Vercel, or straight into a folder on any static host. Just make sure the API URL field points at wherever the backend is deployed, and that the backend's CORS settings (already wide open in `app.py`) allow your frontend's origin.

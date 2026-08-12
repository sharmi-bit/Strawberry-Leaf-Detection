"""
FastAPI backend for the strawberry leaf disease detection pipeline.

Endpoints:
  GET  /health          -> liveness check + whether models are loaded
  POST /predict          -> upload an image, get back the predicted class,
                             per-class probabilities, and a base64 PNG of
                             the leaf with the lesion mask overlaid

Run:
    uvicorn app:app --reload --port 8000
"""

import base64
import io
import json
import os

import cv2
import numpy as np
import torch
import xgboost as xgb
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

from models.unet import UNet
from models.feature_extraction import extract_features

BASE_DIR = os.path.dirname(__file__)
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")
UNET_PATH = os.path.join(WEIGHTS_DIR, "unet_leaf_segmentation.pth")
XGB_PATH = os.path.join(WEIGHTS_DIR, "xgboost_classifier.json")
META_PATH = os.path.join(WEIGHTS_DIR, "model_meta.json")

IMG_SIZE = 96  # must match generate_synthetic_data.IMG_SIZE / training input size

app = FastAPI(title="Strawberry Leaf Disease Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

device = "cuda" if torch.cuda.is_available() else "cpu"

unet_model = None
xgb_model = None
classes = None


def load_models():
    global unet_model, xgb_model, classes

    if not (os.path.exists(UNET_PATH) and os.path.exists(XGB_PATH) and os.path.exists(META_PATH)):
        return False

    with open(META_PATH) as f:
        meta = json.load(f)
    classes = meta["classes"]
    base_ch = meta.get("unet_base_ch", 16)

    unet_model = UNet(in_channels=3, out_channels=1, base_ch=base_ch).to(device)
    unet_model.load_state_dict(torch.load(UNET_PATH, map_location=device))
    unet_model.eval()

    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(XGB_PATH)

    return True


MODELS_READY = load_models()


def preprocess_image(pil_img: Image.Image):
    pil_img = pil_img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    rgb = np.array(pil_img)
    return rgb


def looks_like_leaf(rgb: np.ndarray, min_green_fraction: float = 0.12) -> bool:
    """
    Cheap out-of-distribution guard: flags images that are very unlikely to
    be a leaf photo at all (e.g. a person, a document, a random object), so
    the classifier doesn't confidently force a diagnosis onto them.

    Not a trained model - just checks what fraction of pixels fall in a
    plant-plausible green/yellow-green hue range. A real leaf, even a
    heavily diseased one, still has enough green tissue to clear a low bar;
    skin tones, fabric, walls, etc. generally don't.
    """
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    # green/yellow-green hue band (OpenCV hue range 0-179), with a minimum
    # saturation/value so dark or washed-out pixels don't count either way
    green_like = (h >= 25) & (h <= 95) & (s >= 25) & (v >= 25)
    fraction = float(green_like.mean())
    return fraction >= min_green_fraction


def run_pipeline(rgb: np.ndarray):
    # --- segmentation ---
    tensor = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = unet_model(tensor)
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()
    mask = (probs > 0.5).astype(np.uint8)

    # --- feature extraction + classification ---
    feats = extract_features(rgb, mask).reshape(1, -1)
    class_probs = xgb_model.predict_proba(feats)[0]
    pred_idx = int(np.argmax(class_probs))

    # --- overlay mask on image for visualization ---
    overlay = rgb.copy()
    red_layer = np.zeros_like(overlay)
    red_layer[..., 0] = 255
    alpha = 0.45
    mask_bool = mask.astype(bool)
    overlay[mask_bool] = (
        overlay[mask_bool] * (1 - alpha) + red_layer[mask_bool] * alpha
    ).astype(np.uint8)

    overlay_img = Image.fromarray(overlay)
    buf = io.BytesIO()
    overlay_img.save(buf, format="PNG")
    overlay_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    lesion_area_pct = float(mask.mean() * 100)

    return {
        "predicted_class": classes[pred_idx],
        "confidence": float(class_probs[pred_idx]),
        "class_probabilities": {c: float(p) for c, p in zip(classes, class_probs)},
        "lesion_area_percent": round(lesion_area_pct, 2),
        "overlay_image_base64": overlay_b64,
    }


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": MODELS_READY, "device": device}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not MODELS_READY:
        raise HTTPException(
            status_code=503,
            detail="Model weights not found. Run train_unet.py and train_xgboost.py first.",
        )
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg", "image/webp"):
        raise HTTPException(status_code=400, detail="Please upload a JPEG, PNG, or WEBP image.")

    contents = await file.read()
    try:
        pil_img = Image.open(io.BytesIO(contents))
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image file.")

    rgb = preprocess_image(pil_img)

    if not looks_like_leaf(rgb):
        raise HTTPException(
            status_code=422,
            detail="This doesn't look like a leaf photo. Try uploading a clear photo of a strawberry leaf.",
        )

    result = run_pipeline(rgb)
    return JSONResponse(result)

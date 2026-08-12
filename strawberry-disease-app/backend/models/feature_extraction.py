"""
Feature extraction pipeline that turns a (RGB image, lesion mask) pair into a
fixed-length numeric feature vector for the XGBoost classifier.

Feature groups:
  1. Lesion geometry   - area ratio, region count, perimeter/area ratio
  2. Colour statistics - RGB + HSV mean/std, both whole-leaf and lesion-only
  3. Texture (GLCM)    - contrast, homogeneity, energy, correlation on the
                          grayscale lesion region
"""

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import label, regionprops

FEATURE_NAMES = [
    "lesion_area_ratio",
    "lesion_region_count",
    "lesion_perimeter_area_ratio",
    "leaf_mean_r", "leaf_mean_g", "leaf_mean_b",
    "leaf_std_r", "leaf_std_g", "leaf_std_b",
    "leaf_mean_h", "leaf_mean_s", "leaf_mean_v",
    "lesion_mean_r", "lesion_mean_g", "lesion_mean_b",
    "lesion_std_r", "lesion_std_g", "lesion_std_b",
    "lesion_mean_h", "lesion_mean_s", "lesion_mean_v",
    "glcm_contrast", "glcm_homogeneity", "glcm_energy", "glcm_correlation",
]


def _safe_mean_std(arr):
    if arr.size == 0:
        return 0.0, 0.0
    return float(arr.mean()), float(arr.std())


def extract_features(rgb_image: np.ndarray, lesion_mask: np.ndarray) -> np.ndarray:
    """
    rgb_image:   HxWx3 uint8 RGB image
    lesion_mask: HxW binary mask (1 = lesion / diseased pixel, 0 = healthy)
    Returns: 1D float32 feature vector matching FEATURE_NAMES
    """
    mask = lesion_mask.astype(bool)
    h, w = mask.shape
    total_px = h * w

    # --- geometry ---
    lesion_area_ratio = mask.sum() / total_px
    labeled = label(mask)
    regions = regionprops(labeled)
    region_count = len(regions)
    if region_count > 0:
        perim = sum(r.perimeter for r in regions)
        area = sum(r.area for r in regions)
        perim_area_ratio = perim / area if area > 0 else 0.0
    else:
        perim_area_ratio = 0.0

    # --- colour: whole leaf ---
    r, g, b = rgb_image[..., 0], rgb_image[..., 1], rgb_image[..., 2]
    leaf_mean_r, leaf_std_r = _safe_mean_std(r.astype(np.float32))
    leaf_mean_g, leaf_std_g = _safe_mean_std(g.astype(np.float32))
    leaf_mean_b, leaf_std_b = _safe_mean_std(b.astype(np.float32))

    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV).astype(np.float32)
    leaf_mean_h = float(hsv[..., 0].mean())
    leaf_mean_s = float(hsv[..., 1].mean())
    leaf_mean_v = float(hsv[..., 2].mean())

    # --- colour: lesion region only ---
    if mask.sum() > 0:
        lesion_mean_r, lesion_std_r = _safe_mean_std(r[mask].astype(np.float32))
        lesion_mean_g, lesion_std_g = _safe_mean_std(g[mask].astype(np.float32))
        lesion_mean_b, lesion_std_b = _safe_mean_std(b[mask].astype(np.float32))
        lesion_mean_h = float(hsv[..., 0][mask].mean())
        lesion_mean_s = float(hsv[..., 1][mask].mean())
        lesion_mean_v = float(hsv[..., 2][mask].mean())
    else:
        lesion_mean_r = lesion_std_r = 0.0
        lesion_mean_g = lesion_std_g = 0.0
        lesion_mean_b = lesion_std_b = 0.0
        lesion_mean_h = lesion_mean_s = lesion_mean_v = 0.0

    # --- texture: GLCM on grayscale, masked to lesion bounding box ---
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    if mask.sum() > 20:
        ys, xs = np.where(mask)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        patch = gray[y0:y1, x0:x1]
        patch = (patch // 32).astype(np.uint8)  # quantize to 8 levels for stability
        glcm = graycomatrix(patch, distances=[1], angles=[0], levels=8, symmetric=True, normed=True)
        contrast = float(graycoprops(glcm, "contrast")[0, 0])
        homogeneity = float(graycoprops(glcm, "homogeneity")[0, 0])
        energy = float(graycoprops(glcm, "energy")[0, 0])
        correlation = float(graycoprops(glcm, "correlation")[0, 0])
        if np.isnan(correlation):
            correlation = 0.0
    else:
        contrast = homogeneity = energy = correlation = 0.0

    features = np.array([
        lesion_area_ratio, region_count, perim_area_ratio,
        leaf_mean_r, leaf_mean_g, leaf_mean_b,
        leaf_std_r, leaf_std_g, leaf_std_b,
        leaf_mean_h, leaf_mean_s, leaf_mean_v,
        lesion_mean_r, lesion_mean_g, lesion_mean_b,
        lesion_std_r, lesion_std_g, lesion_std_b,
        lesion_mean_h, lesion_mean_s, lesion_mean_v,
        contrast, homogeneity, energy, correlation,
    ], dtype=np.float32)

    return features

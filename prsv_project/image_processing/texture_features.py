from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

from image_processing.constants import (
    GLCM_ANGLES,
    GLCM_DISTANCES,
    LBP_N_POINTS,
    LBP_RADIUS,
)


@dataclass
class TextureFeatureResult:
    feature_dict: dict[str, float]


def _masked_bounding_box(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Return (y0, y1, x0, x1) bounding box of the nonzero mask region, or None if empty."""
    ys, xs = np.nonzero(mask > 0)
    if ys.size == 0:
        return None
    return int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1


def extract_texture_features(grayscale: np.ndarray, mask: np.ndarray) -> TextureFeatureResult:
    """
    Compute GLCM (gray-level co-occurrence matrix) and LBP (local binary
    pattern) texture features over the masked leaf region.

    Ring-spot lesions create a locally irregular, pitted texture that global
    color/brightness averages don't capture. GLCM contrast/homogeneity/energy/
    correlation and LBP uniformity are computed on the leaf's bounding box
    (background pixels outside the mask are zeroed, which is standard practice
    for masked GLCM/LBP - the surrounding zero region has negligible effect on
    a leaf that fills most of its bounding box).
    """
    bbox = _masked_bounding_box(mask)
    if bbox is None:
        return TextureFeatureResult(
            feature_dict={
                "glcm_contrast": 0.0,
                "glcm_homogeneity": 0.0,
                "glcm_energy": 0.0,
                "glcm_correlation": 0.0,
                "lbp_uniformity": 0.0,
            }
        )

    y0, y1, x0, x1 = bbox
    region_gray = grayscale[y0:y1, x0:x1].copy()
    region_mask = mask[y0:y1, x0:x1]
    region_gray[region_mask == 0] = 0

    # GLCM expects a quantized (fewer gray levels) image for a tractable
    # co-occurrence matrix. Requantize to 32 levels.
    levels = 32
    quantized = (region_gray.astype(np.float32) / 255.0 * (levels - 1)).astype(np.uint8)

    try:
        glcm = graycomatrix(
            quantized,
            distances=GLCM_DISTANCES,
            angles=GLCM_ANGLES,
            levels=levels,
            symmetric=True,
            normed=True,
        )
        glcm_contrast = float(np.mean(graycoprops(glcm, "contrast")))
        glcm_homogeneity = float(np.mean(graycoprops(glcm, "homogeneity")))
        glcm_energy = float(np.mean(graycoprops(glcm, "energy")))
        glcm_correlation = float(np.nan_to_num(np.mean(graycoprops(glcm, "correlation")), nan=0.0))
    except Exception:  # noqa: BLE001 - texture extraction must never break the pipeline
        glcm_contrast = glcm_homogeneity = glcm_energy = glcm_correlation = 0.0

    try:
        lbp = local_binary_pattern(region_gray, P=LBP_N_POINTS, R=LBP_RADIUS, method="uniform")
        valid_lbp = lbp[region_mask > 0]
        if valid_lbp.size:
            n_bins = LBP_N_POINTS + 2
            hist, _ = np.histogram(valid_lbp, bins=n_bins, range=(0, n_bins), density=True)
            # "Uniformity" here = concentration of the LBP histogram (higher = more
            # uniform/regular texture, lower = more irregular/pitted texture -
            # exactly the kind of irregularity ring-spot lesions introduce).
            lbp_uniformity = float(np.sum(hist ** 2))
        else:
            lbp_uniformity = 0.0
    except Exception:  # noqa: BLE001
        lbp_uniformity = 0.0

    # Normalize contrast (unbounded in principle) into a roughly 0-1 range for
    # consistency with the other handcrafted features the SVM already expects.
    glcm_contrast_norm = float(min(glcm_contrast / (levels ** 2), 1.0))

    return TextureFeatureResult(
        feature_dict={
            "glcm_contrast": round(glcm_contrast_norm, 6),
            "glcm_homogeneity": round(glcm_homogeneity, 6),
            "glcm_energy": round(glcm_energy, 6),
            "glcm_correlation": round(max(min(glcm_correlation, 1.0), -1.0), 6),
            "lbp_uniformity": round(lbp_uniformity, 6),
        }
    )

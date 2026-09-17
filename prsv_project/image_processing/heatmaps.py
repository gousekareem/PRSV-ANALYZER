from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from app.utils.image_utils import normalize_to_uint8


@dataclass
class HeatmapResult:
    edge_heatmap_bgr: np.ndarray
    edge_overlay_rgb: np.ndarray
    severity_heatmap_bgr: np.ndarray
    severity_overlay_rgb: np.ndarray


def _colorize_heatmap(heatmap_gray: np.ndarray) -> np.ndarray:
    """
    Convert grayscale heatmap to color BGR heatmap.
    """
    heatmap_uint8 = normalize_to_uint8(heatmap_gray)
    return cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)


def _overlay_on_rgb(base_rgb: np.ndarray, heatmap_bgr: np.ndarray, alpha: float = 0.38) -> np.ndarray:
    """
    Overlay BGR heatmap on RGB base image and return RGB result.
    """
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(base_rgb, 1.0 - alpha, heatmap_rgb, alpha, 0)


def generate_heatmaps(
    image_rgb: np.ndarray,
    edge_map: np.ndarray,
    gradient_magnitude: np.ndarray,
    laplacian_response: np.ndarray,
    symptom_mask: np.ndarray,
    leaf_mask: np.ndarray,
) -> HeatmapResult:
    """
    Generate paper/demo-quality explainability heatmaps.
    """
    masked_edge = cv2.bitwise_and(edge_map, edge_map, mask=leaf_mask)

    severity_map = (
        0.25 * edge_map.astype(np.float32)
        + 0.30 * gradient_magnitude.astype(np.float32)
        + 0.20 * laplacian_response.astype(np.float32)
        + 0.25 * symptom_mask.astype(np.float32)
    )
    severity_map = severity_map * (leaf_mask > 0)

    edge_heatmap_bgr = _colorize_heatmap(masked_edge)
    severity_heatmap_bgr = _colorize_heatmap(severity_map)

    edge_overlay_rgb = _overlay_on_rgb(image_rgb, edge_heatmap_bgr, alpha=0.35)
    severity_overlay_rgb = _overlay_on_rgb(image_rgb, severity_heatmap_bgr, alpha=0.40)

    edge_overlay_rgb = cv2.bitwise_and(edge_overlay_rgb, edge_overlay_rgb, mask=leaf_mask)
    severity_overlay_rgb = cv2.bitwise_and(severity_overlay_rgb, severity_overlay_rgb, mask=leaf_mask)

    return HeatmapResult(
        edge_heatmap_bgr=edge_heatmap_bgr,
        edge_overlay_rgb=edge_overlay_rgb,
        severity_heatmap_bgr=severity_heatmap_bgr,
        severity_overlay_rgb=severity_overlay_rgb,
    )
from __future__ import annotations

"""
Partial Dependence / Individual Conditional Expectation curves (v3.1):
shows how each feature affects prediction probability across its full
range, complementing SHAP/LIME's single-instance explanations with a view
of the model's behavior across the whole feature range.
"""

from typing import Dict, List

import numpy as np
from sklearn.inspection import partial_dependence


def compute_pdp_ice(model, X: np.ndarray, feature_index: int, feature_name: str, grid_resolution: int = 20) -> Dict[str, object]:
    result = partial_dependence(
        model, X, features=[feature_index], grid_resolution=grid_resolution, kind="both"
    )

    return {
        "feature_name": feature_name,
        "grid_values": [round(float(v), 6) for v in result["grid_values"][0]],
        "average_pdp": [round(float(v), 6) for v in result["average"][0]],
        "individual_ice_curves": [[round(float(v), 6) for v in curve] for curve in result["individual"][0]],
    }


def compute_pdp_for_all_features(model, X: np.ndarray, feature_names: List[str], grid_resolution: int = 20) -> List[Dict[str, object]]:
    return [
        compute_pdp_ice(model, X, feature_index=i, feature_name=name, grid_resolution=grid_resolution)
        for i, name in enumerate(feature_names)
    ]

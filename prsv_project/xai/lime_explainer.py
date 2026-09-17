from __future__ import annotations

"""
LIME explanations (v3.0), complementing the existing SHAP explainer
(ml/shap_explainer.py).

Scope note, stated plainly: the upgrade wishlist describes "LIME on image
superpixels." This pipeline is deliberately NOT a pixel-level CNN - it
classifies from 12 hand-engineered numeric features (green_ratio,
edge_density, entropy, etc. - see ml/feature_schema.py), which is the whole
point of the project's "lightweight, explainable-by-design, non-deep-learning"
approach (see Section 14's Green-AI framing). Superpixel-LIME requires a
model that consumes raw image pixels; applying it here would need bolting on
a separate pixel classifier no one asked to add, which would quietly change
the project's core design decision. What genuinely applies to *this*
architecture is LIME's tabular mode (lime.lime_tabular), which perturbs the
12 engineered features around an instance and fits a local linear model -
exactly the same "why this instance" question, answered in the feature space
the classifier actually sees. That is what's implemented below; SHAP already
covers global feature attribution, LIME adds a second, model-agnostic
cross-check on the *local* explanation for one specific prediction.
"""

from typing import Dict, Optional

import numpy as np

from app.config import Settings
from ml.feature_schema import EXPECTED_FEATURE_NAMES
from ml.model_loader import load_model_artifacts

_lime_explainer_cache: dict = {}


def _get_training_reference(settings: Settings) -> Optional[np.ndarray]:
    """
    LIME needs a background distribution of feature values to know realistic
    perturbation ranges. Reuses the same SHAP k-means background summary
    already computed at training time (ml/train_svm.py) rather than
    requiring a second artifact.
    """
    try:
        import joblib

        if not settings.shap_background_path.exists():
            return None
        background = joblib.load(settings.shap_background_path)
        return np.asarray(background.data) if hasattr(background, "data") else np.asarray(background)
    except Exception:  # noqa: BLE001
        return None


def _build_lime_explainer(settings: Settings):
    try:
        from lime.lime_tabular import LimeTabularExplainer

        reference = _get_training_reference(settings)
        if reference is None:
            return None

        return LimeTabularExplainer(
            training_data=reference,
            feature_names=EXPECTED_FEATURE_NAMES,
            class_names=["Healthy", "Diseased"],
            mode="classification",
            discretize_continuous=True,
        )
    except Exception:  # noqa: BLE001 - LIME is an optional, best-effort explainer
        return None


def _get_cached_explainer(settings: Settings):
    key = str(settings.models_dir)
    if key not in _lime_explainer_cache:
        _lime_explainer_cache[key] = _build_lime_explainer(settings)
    return _lime_explainer_cache[key]


def explain_with_lime(
    feature_vector: list[float],
    settings: Settings,
    num_features: int = 6,
) -> Optional[Dict[str, float]]:
    """
    Return LIME's local per-feature contribution weights for one prediction,
    or None if LIME/the background reference isn't available - same
    best-effort, never-fatal pattern as ml/shap_explainer.py.
    """
    explainer = _get_cached_explainer(settings)
    if explainer is None:
        return None

    artifacts = load_model_artifacts(settings)
    if not artifacts.model_available or artifacts.model is None or artifacts.scaler is None:
        return None

    try:
        features = np.array(feature_vector, dtype=np.float32).reshape(1, -1)
        scaled = artifacts.scaler.transform(features)[0]

        def predict_fn(X: np.ndarray) -> np.ndarray:
            return artifacts.model.predict_proba(X)

        explanation = explainer.explain_instance(
            scaled,
            predict_fn,
            num_features=min(num_features, len(EXPECTED_FEATURE_NAMES)),
        )

        return {feature_desc: round(float(weight), 6) for feature_desc, weight in explanation.as_list()}
    except Exception:  # noqa: BLE001
        return None

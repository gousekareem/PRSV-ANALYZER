from __future__ import annotations

from typing import Optional

import numpy as np

from app.config import Settings
from ml.feature_schema import EXPECTED_FEATURE_NAMES
from ml.model_loader import load_model_artifacts

_explainer_state_cache: dict = {}


def _load_explainer_state(settings: Settings):
    """
    Cached (module-level dict, not functools.lru_cache - Settings/pydantic
    objects aren't hashable) so the relatively expensive KernelExplainer
    background setup only happens once per process, not once per image.
    """
    cache_key = str(settings.models_dir)
    if cache_key in _explainer_state_cache:
        return _explainer_state_cache[cache_key]

    state = _build_explainer_state(settings)
    _explainer_state_cache[cache_key] = state
    return state


def _build_explainer_state(settings: Settings):
    artifacts = load_model_artifacts(settings)
    if not artifacts.model_available or artifacts.model is None or artifacts.scaler is None:
        return None

    if not settings.shap_background_path.exists():
        return None

    try:
        import joblib
        import shap

        background = joblib.load(settings.shap_background_path)

        def _predict_proba(X: np.ndarray) -> np.ndarray:
            return artifacts.model.predict_proba(X)

        explainer = shap.KernelExplainer(_predict_proba, background)
        return {"explainer": explainer, "scaler": artifacts.scaler, "model": artifacts.model}
    except Exception:  # noqa: BLE001 - explainability is best-effort, never fatal
        return None


def explain_prediction(
    feature_vector: list[float],
    prediction: str,
    settings: Settings,
) -> Optional[dict[str, float]]:
    """
    Return per-feature SHAP contribution toward the predicted class, or None
    if SHAP isn't available/couldn't run (heuristic-fallback mode, missing
    background sample, or any runtime error). Contributions are signed:
    positive = pushed the prediction toward the predicted class, negative =
    pushed away from it.
    """
    state = _load_explainer_state(settings)
    if state is None:
        return None

    try:
        features = np.array(feature_vector, dtype=np.float32).reshape(1, -1)
        scaled = state["scaler"].transform(features)

        # nsamples kept small and fixed (rather than SHAP's "auto", which can
        # run into the thousands) so a single image's explanation stays fast
        # enough for a synchronous request - this is an approximation, not an
        # exact Shapley value, which is an accepted trade-off for interactive use.
        shap_values = state["explainer"].shap_values(scaled, nsamples=200, silent=True)

        model = state["model"]
        class_index = 0
        if hasattr(model, "classes_"):
            classes = list(model.classes_)
            predicted_class_label = 1 if prediction.strip().lower() != "healthy" else 0
            if predicted_class_label in classes:
                class_index = classes.index(predicted_class_label)
            elif len(classes) > 1:
                class_index = min(1, len(classes) - 1)

        if isinstance(shap_values, list):
            class_shap = shap_values[class_index][0]
        else:
            # Newer SHAP versions may return a single (1, n_features, n_classes) array
            class_shap = np.asarray(shap_values)
            if class_shap.ndim == 3:
                class_shap = class_shap[0, :, class_index]
            else:
                class_shap = class_shap[0]

        return {
            name: round(float(value), 6)
            for name, value in zip(EXPECTED_FEATURE_NAMES, class_shap)
        }
    except Exception:  # noqa: BLE001 - explainability is best-effort, never fatal
        return None

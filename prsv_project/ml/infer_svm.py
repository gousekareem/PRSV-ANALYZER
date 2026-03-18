from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from app.config import Settings
from ml.feature_schema import EXPECTED_FEATURE_DIM, EXPECTED_FEATURE_NAMES
from ml.model_loader import LoadedModelArtifacts, load_model_artifacts


@dataclass
class InferenceResult:
    prediction: str
    confidence: float
    probabilities: dict[str, float]
    inference_mode: str
    feature_vector_used: list[float]


def _validate_feature_vector(feature_vector: list[float]) -> np.ndarray:
    if len(feature_vector) != EXPECTED_FEATURE_DIM:
        raise ValueError(
            f"Feature vector length mismatch. Expected {EXPECTED_FEATURE_DIM}, got {len(feature_vector)}."
        )
    return np.array(feature_vector, dtype=np.float32).reshape(1, -1)


def _heuristic_predict(feature_vector: list[float]) -> InferenceResult:
    """
    Fallback prediction when trained model files are not present.
    This keeps the full pipeline runnable for research demo purposes.
    """
    brightness, green_ratio, hue_mean, saturation_mean, edge_density, color_variance, entropy = feature_vector

    disease_signal = (
        0.30 * (1.0 - green_ratio)
        + 0.25 * edge_density
        + 0.20 * entropy
        + 0.15 * color_variance
        + 0.10 * saturation_mean
    )
    disease_signal = max(0.0, min(1.0, disease_signal))

    if disease_signal >= 0.42:
        prediction = "Diseased"
        diseased_prob = 0.55 + 0.45 * disease_signal
    else:
        prediction = "Healthy"
        diseased_prob = 0.15 + 0.60 * disease_signal

    diseased_prob = float(max(0.0, min(1.0, diseased_prob)))
    healthy_prob = float(1.0 - diseased_prob)

    if prediction == "Healthy":
        confidence = healthy_prob
    else:
        confidence = diseased_prob

    return InferenceResult(
        prediction=prediction,
        confidence=round(confidence, 4),
        probabilities={
            "Healthy": round(healthy_prob, 4),
            "Diseased": round(diseased_prob, 4),
        },
        inference_mode="heuristic_fallback",
        feature_vector_used=[round(float(x), 6) for x in feature_vector],
    )


def _model_predict(artifacts: LoadedModelArtifacts, feature_vector: list[float]) -> InferenceResult:
    features = _validate_feature_vector(feature_vector)
    transformed = artifacts.scaler.transform(features) if artifacts.scaler is not None else features

    raw_prediction = artifacts.model.predict(transformed)[0]

    if artifacts.label_encoder is not None:
        prediction = str(artifacts.label_encoder.inverse_transform([raw_prediction])[0])
    else:
        prediction = str(raw_prediction)

    probabilities: dict[str, float] = {}
    confidence = 0.0

    if hasattr(artifacts.model, "predict_proba"):
        probs = artifacts.model.predict_proba(transformed)[0]
        if hasattr(artifacts.model, "classes_"):
            classes = list(artifacts.model.classes_)
            if artifacts.label_encoder is not None:
                decoded_classes = [
                    str(artifacts.label_encoder.inverse_transform([cls])[0]) for cls in classes
                ]
            else:
                decoded_classes = [str(cls) for cls in classes]

            probabilities = {
                decoded_classes[i]: round(float(probs[i]), 4) for i in range(len(decoded_classes))
            }
            confidence = float(max(probs))
    else:
        confidence = 0.75

    return InferenceResult(
        prediction=prediction,
        confidence=round(confidence, 4),
        probabilities=probabilities,
        inference_mode="trained_model",
        feature_vector_used=[round(float(x), 6) for x in feature_vector],
    )


def predict_with_svm(feature_vector: list[float], settings: Settings) -> InferenceResult:
    """
    Use trained SVM artifacts if available; otherwise fall back to heuristic inference.
    """
    artifacts = load_model_artifacts(settings)
    if artifacts.model_available and artifacts.model is not None:
        return _model_predict(artifacts, feature_vector)

    return _heuristic_predict(feature_vector)
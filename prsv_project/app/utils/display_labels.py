from __future__ import annotations


def prediction_display_label(prediction: str) -> str:
    normalized = (prediction or "").strip().lower()

    if normalized == "healthy":
        return "Healthy"

    if normalized == "diseased":
        return "PRSV Suspected"

    return prediction or "Unknown"


def prediction_support_text(prediction: str) -> str:
    normalized = (prediction or "").strip().lower()

    if normalized == "healthy":
        return "No strong PRSV-like symptom pattern detected."
    if normalized == "diseased":
        return "PRSV-like symptom pattern detected."
    return "Prediction available."
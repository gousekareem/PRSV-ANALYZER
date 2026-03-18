from __future__ import annotations

from typing import Dict, List

from app.utils.display_labels import prediction_display_label
from rag.schemas import RetrievedChunk


def _compose_evidence_summary(retrieved_chunks: List[RetrievedChunk]) -> str:
    if not retrieved_chunks:
        return "No supporting PRSV evidence was retrieved."
    return " ".join(chunk.text for chunk in retrieved_chunks[:2])


def generate_technical_explanation(
    prediction: str,
    confidence: float,
    severity_label: str,
    severity_score: float,
    feature_values: Dict[str, float],
    retrieved_chunks: List[RetrievedChunk],
) -> str:
    evidence_summary = _compose_evidence_summary(retrieved_chunks)
    display_prediction = prediction_display_label(prediction)

    if prediction.strip().lower() == "healthy":
        diagnosis_sentence = (
            f"The papaya leaf image was assessed as {display_prediction} with confidence {confidence:.4f}. "
            f"The current image-derived pattern does not show strong PRSV-like symptom evidence."
        )
    else:
        diagnosis_sentence = (
            f"The papaya leaf image was assessed as {display_prediction} with confidence {confidence:.4f}. "
            f"The current image-derived pattern is compatible with PRSV-like symptom evidence."
        )

    return (
        f"{diagnosis_sentence} "
        f"Image-derived evidence indicates green ratio {feature_values.get('green_ratio', 0.0):.4f}, "
        f"edge density {feature_values.get('edge_density', 0.0):.4f}, entropy {feature_values.get('entropy', 0.0):.4f}, "
        f"and saturation mean {feature_values.get('saturation_mean', 0.0):.4f}. "
        f"The estimated severity is {severity_label} with score {severity_score:.2f}. "
        f"Retrieved PRSV knowledge supports interpretation of discoloration, chlorosis, mosaic-like symptoms, "
        f"and symptom progression patterns. Evidence summary: {evidence_summary}"
    )


def generate_farmer_friendly_explanation(
    prediction: str,
    severity_label: str,
    retrieved_chunks: List[RetrievedChunk],
) -> str:
    evidence_summary = _compose_evidence_summary(retrieved_chunks)
    display_prediction = prediction_display_label(prediction)

    if prediction.strip().lower() == "healthy":
        opening = (
            f"The system marked this papaya leaf as {display_prediction}. "
            f"No strong PRSV-like symptom pattern was detected in the current image."
        )
    else:
        opening = (
            f"The system marked this papaya leaf as {display_prediction} with {severity_label.lower()} symptom intensity. "
            f"The leaf shows visual changes that may be compatible with Papaya Ring Spot Virus."
        )

    return (
        f"{opening} "
        f"Based on stored PRSV reference knowledge, similar symptoms can be associated with virus infection "
        f"and field spread risk. Guidance summary: {evidence_summary}"
    )


def generate_advisory_notes(
    prediction: str,
    severity_label: str,
) -> List[str]:
    notes = [
        "Inspect nearby papaya leaves for similar PRSV-like symptom patterns.",
        "Monitor the field for aphid activity because PRSV can spread through vectors.",
        "Retain this image report for agronomy verification and project documentation.",
    ]

    if prediction.lower() != "healthy":
        notes.append("Isolate or mark PRSV-suspected plants to reduce possible local spread risk.")

    if severity_label.lower() in {"moderate", "moderate to severe", "severe"}:
        notes.append("Seek agricultural expert review quickly if multiple plants show similar PRSV-like symptoms.")

    return notes
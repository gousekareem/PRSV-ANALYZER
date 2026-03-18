from __future__ import annotations

from typing import Dict, List


def build_observation_query(
    prediction: str,
    severity_label: str,
    feature_values: Dict[str, float],
    severity_score: float,
    symptom_findings: Dict[str, float],
) -> str:
    """
    Build a retrieval query from image-derived findings.
    """
    return (
        f"Papaya leaf {prediction} with severity {severity_label}, severity score {severity_score:.2f}, "
        f"green ratio {feature_values.get('green_ratio', 0.0):.4f}, "
        f"edge density {feature_values.get('edge_density', 0.0):.4f}, "
        f"entropy {feature_values.get('entropy', 0.0):.4f}, "
        f"saturation mean {feature_values.get('saturation_mean', 0.0):.4f}, "
        f"abnormal color score {symptom_findings.get('abnormal_color_score', 0.0):.4f}, "
        f"symptom region ratio {symptom_findings.get('symptom_region_ratio', 0.0):.4f}, "
        f"possible chlorosis, mosaic, discoloration, texture irregularity, PRSV symptoms."
    )


def build_key_findings(
    prediction: str,
    confidence: float,
    severity_label: str,
    severity_score: float,
    feature_values: Dict[str, float],
    symptom_findings: Dict[str, float],
    segmentation_success: bool,
) -> List[str]:
    return [
        f"Prediction: {prediction}",
        f"Confidence: {confidence:.4f}",
        f"Severity label: {severity_label}",
        f"Severity score: {severity_score:.2f}",
        f"Green ratio: {feature_values.get('green_ratio', 0.0):.4f}",
        f"Edge density: {feature_values.get('edge_density', 0.0):.4f}",
        f"Entropy: {feature_values.get('entropy', 0.0):.4f}",
        f"Saturation mean: {feature_values.get('saturation_mean', 0.0):.4f}",
        f"Abnormal color score: {symptom_findings.get('abnormal_color_score', 0.0):.4f}",
        f"Symptom region ratio: {symptom_findings.get('symptom_region_ratio', 0.0):.4f}",
        f"Segmentation success: {segmentation_success}",
    ]
from app.config import settings
from app.services.rag_service import RagService


def test_rag_service_returns_evidence() -> None:
    rag_service = RagService(settings)

    trace = rag_service.build_explanation_trace(
        prediction="Diseased",
        confidence=0.88,
        severity_label="Moderate",
        severity_score=42.5,
        feature_values={
            "brightness": 0.4,
            "green_ratio": 0.3,
            "hue_mean": 0.3,
            "saturation_mean": 0.5,
            "edge_density": 0.4,
            "color_variance": 0.2,
            "entropy": 0.6,
        },
        symptom_findings={
            "abnormal_color_score": 0.55,
            "symptom_region_ratio": 0.40,
        },
        segmentation_success=True,
    )

    assert trace.observation_query
    assert len(trace.retrieved_evidence) > 0
    assert trace.technical_explanation
    assert trace.farmer_friendly_explanation
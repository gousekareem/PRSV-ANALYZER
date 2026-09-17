from pathlib import Path

from app.schemas import BatchResult, ExplanationTrace, ImageResult
from app.services.export_service import ExportService


def test_generate_batch_charts(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_test"
    run_dir.mkdir(parents=True, exist_ok=True)

    batch_result = BatchResult(
        run_id="run_test",
        total_images=2,
        processed_images=2,
        failed_images=0,
        healthy_count=1,
        diseased_count=1,
        average_confidence=0.75,
        average_infection_percentage=35.0,
        severity_distribution={"Healthy": 1, "Moderate": 1},
        results=[
            ImageResult(
                image_id="img1",
                filename="a.jpg",
                prediction="Healthy",
                confidence=0.9,
                infection_percentage=0.0,
                severity_score=0.0,
                severity_label="Healthy",
                feature_values={},
                output_paths={},
                explanation_trace=ExplanationTrace(
                    observation_query="q1",
                    key_findings=[],
                    retrieved_evidence=[],
                    technical_explanation="t1",
                    farmer_friendly_explanation="f1",
                    advisory_notes=[],
                ),
            ),
            ImageResult(
                image_id="img2",
                filename="b.jpg",
                prediction="Diseased",
                confidence=0.6,
                infection_percentage=70.0,
                severity_score=70.0,
                severity_label="Moderate",
                feature_values={},
                output_paths={},
                explanation_trace=ExplanationTrace(
                    observation_query="q2",
                    key_findings=[],
                    retrieved_evidence=[],
                    technical_explanation="t2",
                    farmer_friendly_explanation="f2",
                    advisory_notes=[],
                ),
            ),
        ],
        failures=[],
    )

    service = ExportService()
    chart_paths = service.generate_batch_charts(run_dir, batch_result)

    assert Path(chart_paths["prediction_distribution"]).exists()
    assert Path(chart_paths["severity_distribution"]).exists()
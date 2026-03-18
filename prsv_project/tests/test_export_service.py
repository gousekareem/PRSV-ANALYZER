from pathlib import Path

from app.schemas import BatchResult, ExplanationTrace, ImageResult
from app.services.export_service import ExportService


def test_export_service_creates_csv_and_zip(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_test"
    run_dir.mkdir(parents=True, exist_ok=True)

    result = BatchResult(
        run_id="run_test",
        total_images=1,
        processed_images=1,
        failed_images=0,
        healthy_count=1,
        diseased_count=0,
        average_confidence=0.9,
        average_infection_percentage=2.0,
        severity_distribution={"Very mild": 1},
        results=[
            ImageResult(
                image_id="img_001",
                filename="leaf.jpg",
                prediction="Healthy",
                confidence=0.9,
                infection_percentage=2.0,
                severity_score=2.0,
                severity_label="Very mild",
                feature_values={"brightness": 0.5},
                output_paths={"original": "x"},
                explanation_trace=ExplanationTrace(
                    observation_query="q",
                    key_findings=["a"],
                    retrieved_evidence=[],
                    technical_explanation="tech",
                    farmer_friendly_explanation="farm",
                    advisory_notes=["note"],
                ),
            )
        ],
        failures=[],
    )

    exporter = ExportService()
    csv_path = exporter.export_batch_csv(run_dir, result)
    zip_path = exporter.create_run_bundle(run_dir)

    assert csv_path.exists()
    assert zip_path.exists()
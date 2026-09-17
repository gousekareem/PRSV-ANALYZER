from pathlib import Path

import cv2
import numpy as np

from app.config import settings
from app.services.analysis_service import AnalysisService
from app.services.run_manager import RunManager


def test_analysis_service_generates_outputs(tmp_path: Path) -> None:
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    image[:, :] = [20, 180, 20]

    image_path = tmp_path / "sample_leaf.jpg"
    cv2.imwrite(str(image_path), image)

    run_manager = RunManager(settings)
    service = AnalysisService(settings, run_manager)

    result = service.analyze_single_image(image_path)

    assert result.filename == "sample_leaf.jpg"
    assert result.prediction in {"Healthy", "Diseased"}
    assert "preprocessed" in result.output_paths
    assert Path(result.output_paths["preprocessed"]).exists()
    assert Path(result.output_paths["features_json"]).exists()
    assert Path(result.output_paths["prediction_json"]).exists()
    assert Path(result.output_paths["severity_json"]).exists()
    assert Path(result.output_paths["explanation_json"]).exists()
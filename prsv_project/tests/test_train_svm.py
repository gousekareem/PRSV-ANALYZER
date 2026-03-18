from pathlib import Path

import pandas as pd

from ml.train_svm import train_svm_from_feature_csv


def test_train_svm_from_feature_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "training_features.csv"

    df = pd.DataFrame(
        [
            {
                "filename": "a.jpg",
                "label": "Healthy",
                "brightness": 0.5,
                "green_ratio": 0.7,
                "hue_mean": 0.2,
                "saturation_mean": 0.4,
                "edge_density": 0.1,
                "color_variance": 0.1,
                "entropy": 0.2,
            },
            {
                "filename": "b.jpg",
                "label": "Healthy",
                "brightness": 0.52,
                "green_ratio": 0.68,
                "hue_mean": 0.22,
                "saturation_mean": 0.42,
                "edge_density": 0.09,
                "color_variance": 0.11,
                "entropy": 0.21,
            },
            {
                "filename": "c.jpg",
                "label": "Diseased",
                "brightness": 0.35,
                "green_ratio": 0.30,
                "hue_mean": 0.40,
                "saturation_mean": 0.60,
                "edge_density": 0.45,
                "color_variance": 0.35,
                "entropy": 0.60,
            },
            {
                "filename": "d.jpg",
                "label": "Diseased",
                "brightness": 0.36,
                "green_ratio": 0.28,
                "hue_mean": 0.42,
                "saturation_mean": 0.62,
                "edge_density": 0.48,
                "color_variance": 0.36,
                "entropy": 0.58,
            },
        ]
    )
    df.to_csv(csv_path, index=False)

    model_dir = tmp_path / "models"
    summary = train_svm_from_feature_csv(csv_path, model_dir, test_size=0.5, random_state=42)

    assert (model_dir / "svm_model.joblib").exists()
    assert (model_dir / "scaler.joblib").exists()
    assert (model_dir / "label_encoder.joblib").exists()
    assert (model_dir / "metadata.json").exists()
    assert (model_dir / "training_summary.json").exists()
    assert "metrics" in summary
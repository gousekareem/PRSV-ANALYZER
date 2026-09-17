"""
End-to-end retrain: labeled images -> features -> trained SVM artifacts.

Previously this script only re-extracted features into
data/training_features.csv and stopped -- it never actually called the
trainer, so running it did not produce svm_model.joblib. It now:

  1. Extracts handcrafted features for every labeled row in labels_template.csv
  2. Writes data/training_features.csv (kept for inspection/debugging)
  3. Trains an SVM (with cross-validated hyperparameter search) on those
     features via ml.train_svm.train_svm_from_feature_csv
  4. Saves models/svm_model.joblib, scaler.joblib, label_encoder.joblib,
     metadata.json, training_summary.json, and evaluation plots

Usage (from prsv_project/):
    python scripts/auto_label_from_filenames.py   # if labels aren't filled yet
    python scripts/retrain_model.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from app.config import settings
from app.utils.image_utils import read_image_cv
from image_processing.feature_extraction import extract_handcrafted_features
from image_processing.preprocess import preprocess_image
from image_processing.segmentation import segment_leaf
from image_processing.symptom_enhancement import enhance_symptoms
from ml.train_svm import train_svm_from_feature_csv


def extract_features_for_labeled_images() -> Path:
    labels_path = PROJECT_ROOT / "labels_template.csv"
    if not labels_path.exists():
        raise FileNotFoundError(
            "labels_template.csv not found. Run scripts/auto_label_from_filenames.py "
            "(or scripts/create_label_template.py + manual labeling) first."
        )

    labels_df = pd.read_csv(labels_path)
    if "filename" not in labels_df.columns or "label" not in labels_df.columns:
        raise ValueError("labels_template.csv must contain filename and label columns.")

    records = []
    skipped_missing = 0
    skipped_unreadable = 0

    for _, row in labels_df.iterrows():
        filename = str(row["filename"])
        label = str(row["label"]).strip()

        if not label:
            continue

        image_path = settings.demo_dataset_path / filename
        if not image_path.exists():
            skipped_missing += 1
            continue

        try:
            image_bgr = read_image_cv(image_path)
            preprocess_result = preprocess_image(image_bgr, settings)
            segmentation_result = segment_leaf(preprocess_result.enhanced_rgb)
            symptom_result = enhance_symptoms(preprocess_result.enhanced_rgb, segmentation_result.mask)

            feature_result = extract_handcrafted_features(
                image_rgb=preprocess_result.enhanced_rgb,
                grayscale=preprocess_result.grayscale,
                hsv=preprocess_result.hsv,
                edge_map=symptom_result.edge_map,
                mask=segmentation_result.mask,
            )
        except Exception as exc:  # noqa: BLE001 - keep batch extraction going
            print(f"[WARN] Skipping unreadable/failed image {filename}: {exc}")
            skipped_unreadable += 1
            continue

        record = {"filename": filename, "label": label}
        record.update(feature_result.feature_dict)
        records.append(record)

    output_path = PROJECT_ROOT / "data" / "training_features.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)

    print(f"Training feature CSV created: {output_path.resolve()}")
    print(f"Rows written: {len(df)}")
    if skipped_missing:
        print(f"Skipped (file missing on disk): {skipped_missing}")
    if skipped_unreadable:
        print(f"Skipped (unreadable/failed processing): {skipped_unreadable}")

    return output_path


def main() -> None:
    feature_csv_path = extract_features_for_labeled_images()

    model_output_dir = settings.models_dir
    print(f"\nTraining SVM from {feature_csv_path} ...")
    summary = train_svm_from_feature_csv(
        feature_csv_path=feature_csv_path,
        model_output_dir=model_output_dir,
    )

    print(f"\nModel artifacts saved to: {model_output_dir.resolve()}")
    print("  - svm_model.joblib")
    print("  - scaler.joblib")
    print("  - label_encoder.joblib")
    print("  - metadata.json")
    print("  - training_summary.json")
    print("  - evaluation/ (confusion matrix, ROC, PR curve)")

    cv = summary.get("cross_validation", {})
    if cv.get("best_params") is not None:
        print(f"\nBest hyperparameters (cross-validated): {cv['best_params']}")
        if cv.get("best_cv_score") is not None:
            print(f"Best CV f1_macro score: {cv['best_cv_score']:.4f}")

    metrics = summary.get("metrics", {})
    print("\nHeld-out test metrics:")
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.4f}")

    print(
        "\nDone. Restart the app (or it will pick this up on next request) - "
        "inference_mode should now report 'trained_model' instead of 'heuristic_fallback'."
    )


if __name__ == "__main__":
    main()

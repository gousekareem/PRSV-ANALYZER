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


def main() -> None:
    labels_path = PROJECT_ROOT / "labels_template.csv"
    if not labels_path.exists():
        raise FileNotFoundError("labels_template.csv not found. Create it first and fill labels.")

    labels_df = pd.read_csv(labels_path)
    if "filename" not in labels_df.columns or "label" not in labels_df.columns:
        raise ValueError("labels_template.csv must contain filename and label columns.")

    records = []
    for _, row in labels_df.iterrows():
        filename = str(row["filename"])
        label = str(row["label"]).strip()

        if not label:
            continue

        image_path = settings.demo_dataset_path / filename
        if not image_path.exists():
            continue

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

        record = {
            "filename": filename,
            "label": label,
        }
        record.update(feature_result.feature_dict)
        records.append(record)

    output_path = PROJECT_ROOT / "data" / "training_features.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)

    print(f"Training feature CSV created: {output_path.resolve()}")
    print(f"Rows written: {len(df)}")


if __name__ == "__main__":
    main()
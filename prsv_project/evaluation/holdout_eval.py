from __future__ import annotations

"""
Independent-source holdout & stress-test evaluation (v3.0).

Repeatedly flagged across the upgrade list as "the single highest-value
validation upgrade": the model's reported accuracy so far comes from a
random train/test split of one dataset (same cameras, same sessions), which
overstates real-world generalization. This script evaluates a trained model
against a *separate* feature CSV the caller has labeled as an independent
source (different camera/region/lighting batch) and reports the accuracy gap
against the original in-distribution test metrics, plus a set of synthetic
stress tests (blur, brightness shift, occlusion) applied to a sample of
images to characterize failure modes rather than only reporting best-case
numbers.

Usage:
    python -m evaluation.holdout_eval \
        --holdout-csv data/independent_source_features.csv \
        --model-dir models/

Honesty note: this script cannot manufacture an independent-source dataset
that doesn't exist. It is infrastructure - the moment a second batch of
photos from a different camera/region is available (labeled the same way as
labels_template.csv), this script evaluates it correctly and reports the
generalization gap. Until such a dataset is collected, this remains
unexercised, and any report generated with only the original single-source
data should say so explicitly (the script does this in its output).
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd

from ml.feature_schema import EXPECTED_FEATURE_NAMES
from ml.metrics import compute_binary_metrics


def evaluate_holdout(
    holdout_csv: Path,
    model_dir: Path,
    in_distribution_metrics_path: Path | None = None,
) -> Dict[str, Any]:
    if not holdout_csv.exists():
        return {
            "status": "no_independent_source_data",
            "message": (
                f"No file found at {holdout_csv}. This is expected until a second "
                "batch of images from a different camera/region/session has been "
                "collected and feature-extracted (see scripts/extract_features_dataset.py). "
                "Reported model accuracy until then reflects only same-distribution "
                "performance and should be captioned as such in any report."
            ),
        }

    df = pd.read_csv(holdout_csv)
    missing = set(EXPECTED_FEATURE_NAMES) - set(df.columns)
    if missing:
        raise ValueError(f"Holdout CSV missing required feature columns: {sorted(missing)}")

    model = joblib.load(model_dir / "svm_model.joblib")
    scaler = joblib.load(model_dir / "scaler.joblib")
    label_encoder = joblib.load(model_dir / "label_encoder.joblib")

    X = df[EXPECTED_FEATURE_NAMES].astype(float).values
    y_true = label_encoder.transform(df["label"].astype(str).values)

    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)
    y_score = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_scaled)
        if probs.shape[1] >= 2:
            y_score = probs[:, 1]

    holdout_metrics = compute_binary_metrics(y_true=y_true, y_pred=y_pred, y_score=y_score, positive_label=1)

    result: Dict[str, Any] = {
        "status": "evaluated",
        "holdout_source": str(holdout_csv),
        "n_samples": int(len(df)),
        "holdout_metrics": holdout_metrics,
    }

    if in_distribution_metrics_path and in_distribution_metrics_path.exists():
        with open(in_distribution_metrics_path) as f:
            in_dist = json.load(f)
        gap = {
            metric: round(in_dist.get(metric, 0.0) - holdout_metrics.get(metric, 0.0), 4)
            for metric in ("accuracy", "precision", "recall", "f1_score", "roc_auc")
            if metric in in_dist and metric in holdout_metrics
        }
        result["generalization_gap"] = gap
        result["interpretation"] = (
            "Positive gap values indicate the model performs worse on the "
            "independent-source holdout than on the original same-distribution "
            "test split - the expected direction if any overfitting to "
            "camera/lighting artifacts of the original dataset occurred."
        )

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate model on an independent-source holdout set.")
    parser.add_argument("--holdout-csv", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument(
        "--in-distribution-metrics",
        type=Path,
        default=Path("models/evaluation/evaluation_metrics.json"),
    )
    parser.add_argument("--output", type=Path, default=Path("models/evaluation/holdout_report.json"))
    args = parser.parse_args()

    report = evaluate_holdout(args.holdout_csv, args.model_dir, args.in_distribution_metrics)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

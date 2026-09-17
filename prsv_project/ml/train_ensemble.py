from __future__ import annotations

"""
Train the v3.0 stacked ensemble (SVM + Random Forest + Gradient Boosting,
Logistic Regression meta-learner) and a properly cross-validated calibrated
SVM, then statistically compare both against the existing baseline SVM
(ml/train_svm.py's svm_model.joblib) using McNemar's test and calibration
curves/Brier score.

This is a separate script from ml/train_svm.py by design: the baseline SVM
stays the default production model (models/svm_model.joblib, used by
ml/infer_svm.py) unless someone deliberately promotes the ensemble, and both
are kept on disk so the comparison in this script's output
(models/evaluation/ensemble_comparison.json) is always reproducible against
the exact model currently in production.

Usage:
    python -m ml.train_ensemble --feature-csv data/features.csv --model-dir models/
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from app.config import settings as app_settings
from ml.calibration import calibrate_model
from ml.ensemble import train_stacking_ensemble
from ml.feature_schema import EXPECTED_FEATURE_NAMES
from ml.metrics import compute_binary_metrics
from mlops.experiment_tracking import log_metrics, log_params, log_sklearn_model, tracked_run
from evaluation.statistical_tests import compare_models_report
from xai.surrogate_tree import distill_surrogate_tree


def main() -> None:
    parser = argparse.ArgumentParser(description="Train v3.0 calibrated ensemble and compare against baseline SVM.")
    parser.add_argument("--feature-csv", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.feature_csv)
    required_columns = {"filename", "label", *EXPECTED_FEATURE_NAMES}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Feature CSV missing required columns: {sorted(missing)}")

    X = df[EXPECTED_FEATURE_NAMES].astype(float).values
    y_raw = df["label"].astype(str).values

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    with tracked_run(app_settings, run_name="train_ensemble"):
        log_params({"n_train": len(X_train), "n_test": len(X_test), "random_state": args.random_state})

        # 1) Calibrated single SVM (Platt/isotonic via cross-validated folds,
        # not the leaky in-sample SVC(probability=True) path).
        calibration_method = "sigmoid" if len(X_train) < 1000 else "isotonic"
        calibrated_svm = calibrate_model(X_train_scaled, y_train, method=calibration_method)
        calibrated_pred = calibrated_svm.predict(X_test_scaled)
        calibrated_proba = calibrated_svm.predict_proba(X_test_scaled)[:, 1]
        calibrated_metrics = compute_binary_metrics(y_test, calibrated_pred, calibrated_proba)
        log_metrics({f"calibrated_svm_{k}": v for k, v in calibrated_metrics.items() if isinstance(v, (int, float))})

        # 2) Stacked ensemble.
        ensemble_result = train_stacking_ensemble(X_train_scaled, y_train, random_state=args.random_state)
        ensemble_pred = ensemble_result.model.predict(X_test_scaled)
        ensemble_proba = ensemble_result.model.predict_proba(X_test_scaled)[:, 1]
        ensemble_metrics = compute_binary_metrics(y_test, ensemble_pred, ensemble_proba)
        log_metrics({f"ensemble_{k}": v for k, v in ensemble_metrics.items() if isinstance(v, (int, float))})
        log_params(ensemble_result.metadata)
        log_sklearn_model(ensemble_result.model, "stacking_ensemble")

        # 3) Global surrogate decision tree distilled from the ensemble, for
        # a human-readable rule summary of what the ensemble generally does.
        surrogate = distill_surrogate_tree(X_train_scaled, ensemble_result.model, EXPECTED_FEATURE_NAMES)
        log_metrics({"surrogate_tree_fidelity": surrogate.fidelity})

        # 4) Statistical comparison against the existing baseline SVM, if one
        # is already trained and on disk.
        comparison = {"status": "baseline_svm_not_found"}
        baseline_path = args.model_dir / "svm_model.joblib"
        if baseline_path.exists():
            baseline_model = joblib.load(baseline_path)
            baseline_scaler = joblib.load(args.model_dir / "scaler.joblib")
            baseline_X_test_scaled = baseline_scaler.transform(X_test)
            baseline_pred = baseline_model.predict(baseline_X_test_scaled)
            baseline_proba = (
                baseline_model.predict_proba(baseline_X_test_scaled)[:, 1]
                if hasattr(baseline_model, "predict_proba")
                else None
            )
            comparison = compare_models_report(
                y_true=y_test,
                y_pred_a=baseline_pred,
                y_pred_b=ensemble_pred,
                y_prob_a=baseline_proba,
                y_prob_b=ensemble_proba,
                output_dir=args.model_dir / "evaluation",
                name_a="baseline_svm",
                name_b="stacking_ensemble",
            )

    args.model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(ensemble_result.model, args.model_dir / "ensemble_model.joblib")
    joblib.dump(calibrated_svm, args.model_dir / "svm_model_calibrated.joblib")

    evaluation_dir = args.model_dir / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    with open(evaluation_dir / "ensemble_comparison.json", "w") as f:
        json.dump(
            {
                "calibrated_svm_metrics": calibrated_metrics,
                "ensemble_metrics": ensemble_metrics,
                "ensemble_metadata": ensemble_result.metadata,
                "surrogate_tree": {
                    "fidelity": surrogate.fidelity,
                    "feature_importances": surrogate.feature_importances,
                    "rules_text": surrogate.rules_text,
                },
                "baseline_vs_ensemble_comparison": comparison,
            },
            f,
            indent=2,
        )

    print(f"Ensemble accuracy: {ensemble_metrics['accuracy']:.4f} | Baseline comparison: {comparison.get('mcnemar', {})}")
    print(f"Artifacts written to {evaluation_dir / 'ensemble_comparison.json'}")


if __name__ == "__main__":
    main()

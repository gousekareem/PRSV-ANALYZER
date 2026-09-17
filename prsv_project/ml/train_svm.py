from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

from app.utils.json_utils import save_json
from ml.evaluate import evaluate_binary_classifier
from ml.feature_schema import EXPECTED_FEATURE_NAMES

# Hyperparameter grid searched via cross-validation instead of the previous
# fixed C=2.0, gamma='scale'. Kept intentionally small so training stays fast
# on a CPU-only laptop with a few hundred images.
SVM_PARAM_GRID: Dict[str, Any] = {
    "C": [0.5, 1.0, 2.0, 5.0, 10.0],
    "gamma": ["scale", "auto", 0.01, 0.1],
}

CROSS_VALIDATION_FOLDS: int = 5


@dataclass
class TrainingArtifacts:
    model: SVC
    scaler: StandardScaler
    label_encoder: LabelEncoder
    metadata: Dict[str, Any]


def validate_training_dataframe(df: pd.DataFrame) -> None:
    required_columns = {"filename", "label", *EXPECTED_FEATURE_NAMES}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Training data is missing required columns: {sorted(missing)}")

    if df["label"].nunique() < 2:
        raise ValueError("Training dataset must contain at least two classes.")


def train_svm_from_feature_csv(
    feature_csv_path: Path,
    model_output_dir: Path,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Train an RBF SVM from a CSV of extracted features + labels.
    """
    if not feature_csv_path.exists():
        raise FileNotFoundError(f"Feature CSV not found: {feature_csv_path}")

    df = pd.read_csv(feature_csv_path)
    validate_training_dataframe(df)

    X = df[EXPECTED_FEATURE_NAMES].astype(float).values
    y_raw = df["label"].astype(str).values

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    stratify = y if len(np.unique(y)) > 1 else None
    X_train, X_test, y_train, y_test, filenames_train, filenames_test = train_test_split(
        X,
        y,
        df["filename"].values,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Address class imbalance (e.g. 228 Healthy vs 533 Diseased in the shipped
    # dataset) with class_weight='balanced' rather than relying on stratified
    # sampling alone.
    base_model = SVC(
        kernel="rbf",
        probability=True,
        class_weight="balanced",
        random_state=random_state,
    )

    # Cross-validated hyperparameter search over C/gamma, replacing the old
    # fixed C=2.0, gamma='scale'. Folds are stratified so each fold keeps the
    # Healthy/Diseased ratio roughly intact even with a modest dataset size.
    min_class_count = int(np.min(np.bincount(y_train)))
    n_splits = min(CROSS_VALIDATION_FOLDS, min_class_count)

    if n_splits >= 2:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=SVM_PARAM_GRID,
            cv=cv,
            scoring="f1_macro",
            n_jobs=-1,
            refit=True,
        )
        grid_search.fit(X_train_scaled, y_train)

        model = grid_search.best_estimator_
        cv_results = {
            "cv_folds": n_splits,
            "scoring": "f1_macro",
            "best_params": grid_search.best_params_,
            "best_cv_score": float(grid_search.best_score_),
            "mean_test_score_per_candidate": [float(s) for s in grid_search.cv_results_["mean_test_score"]],
            "std_test_score_per_candidate": [float(s) for s in grid_search.cv_results_["std_test_score"]],
            "params_per_candidate": [dict(p) for p in grid_search.cv_results_["params"]],
        }
    else:
        # Too few samples in the smallest class for k-fold CV (e.g. tiny
        # smoke-test datasets). Fall back to a single fit with default
        # hyperparameters rather than erroring out.
        model = base_model.set_params(C=2.0, gamma="scale")
        model.fit(X_train_scaled, y_train)
        cv_results = {
            "cv_folds": 0,
            "scoring": None,
            "best_params": {"C": 2.0, "gamma": "scale"},
            "best_cv_score": None,
            "note": "Cross-validation skipped: fewer than 2 samples in the smallest training class.",
        }

    y_pred = model.predict(X_test_scaled)
    y_score = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test_scaled)
        if probs.shape[1] >= 2:
            y_score = probs[:, 1]

    model_output_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_output_dir / "svm_model.joblib")
    joblib.dump(scaler, model_output_dir / "scaler.joblib")
    joblib.dump(label_encoder, model_output_dir / "label_encoder.joblib")

    # SHAP KernelExplainer needs a small "background" reference sample to
    # estimate feature contributions against. Summarizing the scaled training
    # set down to ~30 representative points (via k-means) keeps per-image SHAP
    # computation fast at inference time (see ml/shap_explainer.py) while still
    # being representative of the training distribution.
    try:
        from shap import kmeans as shap_kmeans

        background_size = min(30, X_train_scaled.shape[0])
        background = shap_kmeans(X_train_scaled, background_size)
        joblib.dump(background, model_output_dir / "shap_background.joblib")
    except Exception:  # noqa: BLE001 - SHAP background is an optional enhancement
        pass

    metadata: Dict[str, Any] = {
        "model_type": "SVM_RBF",
        "feature_names": EXPECTED_FEATURE_NAMES,
        "label_classes": label_encoder.classes_.tolist(),
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "test_filenames": [str(x) for x in filenames_test.tolist()],
        "class_weight": "balanced",
        "hyperparameters": cv_results["best_params"],
        "cross_validation": cv_results,
    }
    save_json(model_output_dir / "metadata.json", metadata)

    evaluation_dir = model_output_dir / "evaluation"
    metrics = evaluate_binary_classifier(
        y_true=y_test,
        y_pred=y_pred,
        y_score=y_score,
        output_dir=evaluation_dir,
        class_labels=label_encoder.classes_.tolist(),
    )

    training_summary = {
        "metadata": metadata,
        "metrics": metrics,
        "cross_validation": cv_results,
    }
    save_json(model_output_dir / "training_summary.json", training_summary)

    return training_summary
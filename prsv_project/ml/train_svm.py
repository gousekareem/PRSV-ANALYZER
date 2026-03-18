from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

from app.utils.json_utils import save_json
from ml.evaluate import evaluate_binary_classifier
from ml.feature_schema import EXPECTED_FEATURE_NAMES


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

    model = SVC(
        kernel="rbf",
        C=2.0,
        gamma="scale",
        probability=True,
        random_state=random_state,
    )
    model.fit(X_train_scaled, y_train)

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

    metadata: Dict[str, Any] = {
        "model_type": "SVM_RBF",
        "feature_names": EXPECTED_FEATURE_NAMES,
        "label_classes": label_encoder.classes_.tolist(),
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "test_filenames": [str(x) for x in filenames_test.tolist()],
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
    }
    save_json(model_output_dir / "training_summary.json", training_summary)

    return training_summary
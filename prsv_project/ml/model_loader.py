from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib

from app.config import Settings


@dataclass
class LoadedModelArtifacts:
    model: Any | None
    scaler: Any | None
    label_encoder: Any | None
    metadata: dict[str, Any]
    model_available: bool


def _safe_load_joblib(path: Path) -> Any | None:
    if not path.exists():
        return None
    return joblib.load(path)


def load_model_artifacts(settings: Settings) -> LoadedModelArtifacts:
    """
    Load trained model artifacts if present.
    """
    model = _safe_load_joblib(settings.model_path)
    scaler = _safe_load_joblib(settings.scaler_path)
    label_encoder = _safe_load_joblib(settings.label_encoder_path)

    metadata: dict[str, Any] = {}
    if settings.model_metadata_path.exists():
        import json

        with settings.model_metadata_path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)

    model_available = model is not None and scaler is not None

    return LoadedModelArtifacts(
        model=model,
        scaler=scaler,
        label_encoder=label_encoder,
        metadata=metadata,
        model_available=model_available,
    )
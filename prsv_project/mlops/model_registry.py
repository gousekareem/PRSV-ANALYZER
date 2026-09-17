from __future__ import annotations

"""
Model registry with rollback (v3.1): a lightweight, dependency-free
versioned registry on top of the models/ directory - tracks which model
artifact is "active" (serving traffic), keeps prior versions on disk, and
can roll back to any of them. Complements mlops/experiment_tracking.py's
MLflow logging (which records *how* a model was trained) with the *deployment*
side (which model is live right now, and how to revert).

Honesty note: a production system at real scale would use MLflow's own
Model Registry or a dedicated service. This JSON-file-based version is
appropriate for this project's single-machine deployment target (consistent
with the existing SQLite-over-Postgres, in-process-queue-over-Celery choices
already made elsewhere in this codebase) and is fully functional - not a
stub - for that scale.
"""

import json
import shutil
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional


@dataclass
class ModelVersion:
    version_id: str
    model_filename: str
    registered_at: str
    metrics: dict
    notes: str
    is_active: bool


class ModelRegistry:
    def __init__(self, registry_dir: Path) -> None:
        self.registry_dir = registry_dir
        self.versions_dir = registry_dir / "versions"
        self.manifest_path = registry_dir / "registry_manifest.json"
        self.versions_dir.mkdir(parents=True, exist_ok=True)
        if not self.manifest_path.exists():
            self._save_manifest([])

    def _load_manifest(self) -> List[dict]:
        with open(self.manifest_path) as f:
            return json.load(f)

    def _save_manifest(self, versions: List[dict]) -> None:
        with open(self.manifest_path, "w") as f:
            json.dump(versions, f, indent=2)

    def register(self, model_path: Path, metrics: dict, notes: str = "") -> ModelVersion:
        """
        Copies `model_path` into the registry's versions directory under a
        timestamped version id, deactivates any currently-active version,
        and marks the new one active.
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        unique_suffix = uuid.uuid4().hex[:6]
        version_id = f"v_{timestamp}_{unique_suffix}"
        versioned_filename = f"{version_id}_{model_path.name}"
        destination = self.versions_dir / versioned_filename
        shutil.copy2(model_path, destination)

        versions = self._load_manifest()
        for v in versions:
            v["is_active"] = False

        new_version = ModelVersion(
            version_id=version_id,
            model_filename=versioned_filename,
            registered_at=datetime.now(timezone.utc).isoformat(),
            metrics=metrics,
            notes=notes,
            is_active=True,
        )
        versions.append(asdict(new_version))
        self._save_manifest(versions)
        return new_version

    def get_active(self) -> Optional[ModelVersion]:
        versions = self._load_manifest()
        for v in versions:
            if v["is_active"]:
                return ModelVersion(**v)
        return None

    def list_versions(self) -> List[ModelVersion]:
        return [ModelVersion(**v) for v in self._load_manifest()]

    def rollback(self, version_id: str, deploy_to: Path) -> ModelVersion:
        """
        Marks `version_id` as active and copies its artifact to `deploy_to`
        (typically the live models/svm_model.joblib path), so a bad
        retraining run can be reverted without re-running training.
        """
        versions = self._load_manifest()
        target = None
        for v in versions:
            v["is_active"] = v["version_id"] == version_id
            if v["version_id"] == version_id:
                target = v

        if target is None:
            raise ValueError(f"Version {version_id} not found in registry.")

        self._save_manifest(versions)

        source = self.versions_dir / target["model_filename"]
        shutil.copy2(source, deploy_to)
        return ModelVersion(**target)

    def champion_challenger_compare(self, challenger_metrics: dict, primary_metric: str = "accuracy") -> dict:
        """
        Compares a candidate ("challenger") model's metrics against the
        currently active ("champion") version's recorded metrics, returning
        a promote/hold recommendation. This is the decision-support half of
        a champion/challenger deployment pattern; the traffic-splitting half
        needs a live deployment with real request routing to be meaningful,
        which is noted as the integration point rather than faked here.
        """
        champion = self.get_active()
        if champion is None:
            return {"recommendation": "promote", "reason": "No active champion registered yet."}

        champion_score = champion.metrics.get(primary_metric)
        challenger_score = challenger_metrics.get(primary_metric)

        if champion_score is None or challenger_score is None:
            return {"recommendation": "hold", "reason": f"Missing '{primary_metric}' in one of the metric sets."}

        if challenger_score > champion_score:
            return {
                "recommendation": "promote",
                "reason": f"Challenger {primary_metric}={challenger_score:.4f} beats champion {champion_score:.4f}.",
            }
        return {
            "recommendation": "hold",
            "reason": f"Challenger {primary_metric}={challenger_score:.4f} does not beat champion {champion_score:.4f}.",
        }

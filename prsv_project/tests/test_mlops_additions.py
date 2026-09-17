import json

import numpy as np

from mlops.drift_detection import detect_feature_drift, summarize_drift
from mlops.model_registry import ModelRegistry


def test_model_registry_register_and_get_active(tmp_path) -> None:
    model_file = tmp_path / "fake_model.joblib"
    model_file.write_text("fake model bytes")

    registry = ModelRegistry(tmp_path / "registry")
    version = registry.register(model_file, metrics={"accuracy": 0.9}, notes="initial")

    active = registry.get_active()
    assert active is not None
    assert active.version_id == version.version_id
    assert active.is_active is True


def test_model_registry_rollback(tmp_path) -> None:
    model_v1 = tmp_path / "model_v1.joblib"
    model_v1.write_text("v1 content")
    model_v2 = tmp_path / "model_v2.joblib"
    model_v2.write_text("v2 content")

    registry = ModelRegistry(tmp_path / "registry")
    v1 = registry.register(model_v1, metrics={"accuracy": 0.85})
    v2 = registry.register(model_v2, metrics={"accuracy": 0.80})  # worse, hypothetically deployed anyway

    assert registry.get_active().version_id == v2.version_id

    deploy_path = tmp_path / "deployed_model.joblib"
    rolled_back = registry.rollback(v1.version_id, deploy_to=deploy_path)

    assert rolled_back.version_id == v1.version_id
    assert registry.get_active().version_id == v1.version_id
    assert deploy_path.read_text() == "v1 content"


def test_champion_challenger_compare_promotes_better_challenger(tmp_path) -> None:
    model_file = tmp_path / "model.joblib"
    model_file.write_text("content")

    registry = ModelRegistry(tmp_path / "registry")
    registry.register(model_file, metrics={"accuracy": 0.80})

    result = registry.champion_challenger_compare({"accuracy": 0.90})
    assert result["recommendation"] == "promote"


def test_detect_feature_drift_flags_shifted_distribution() -> None:
    rng = np.random.default_rng(0)
    reference = {"green_ratio": rng.normal(0.5, 0.1, size=500)}
    current_shifted = {"green_ratio": rng.normal(0.9, 0.1, size=500)}  # clearly shifted

    results = detect_feature_drift(reference, current_shifted)
    assert len(results) == 1
    assert results[0].drift_detected is True

    summary = summarize_drift(results)
    assert summary["overall_status"] == "drift_detected"


def test_detect_feature_drift_no_drift_on_identical_distribution() -> None:
    rng = np.random.default_rng(1)
    reference = {"entropy": rng.normal(0.5, 0.1, size=500)}
    current_same = {"entropy": rng.normal(0.5, 0.1, size=500)}

    results = detect_feature_drift(reference, current_same)
    summary = summarize_drift(results)
    assert summary["overall_status"] == "stable"

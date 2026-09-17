from __future__ import annotations

"""
MLflow experiment tracking wrapper (v3.0).

Used by ml/train_svm.py and ml/train_ensemble.py to log hyperparameters,
metrics, and model artifacts for every training run, giving the "which run
produced which numbers" history the upgrade list flags as missing. Opt-in
via settings.mlflow_tracking_uri (blank = disabled): when disabled, this
becomes a no-op context manager so training scripts run identically with or
without an MLflow server configured (default: a local ./mlruns directory
requires no server at all - "disabled" here really only guards against
`mlflow` not being installed).
"""

from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional

from app.config import Settings


def is_available() -> bool:
    try:
        import mlflow  # noqa: F401

        return True
    except Exception:  # noqa: BLE001
        return False


@contextmanager
def tracked_run(settings: Settings, run_name: str) -> Iterator[Optional["mlflow.ActiveRun"]]:  # type: ignore[name-defined]
    """
    Context manager wrapping a training run in an MLflow run when the
    `mlflow` package is installed; otherwise yields None and callers should
    skip logging calls (or use log_params/log_metrics below, which are
    themselves no-ops when tracking isn't available).
    """
    if not is_available():
        yield None
        return

    import mlflow

    if settings.mlflow_tracking_uri:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    with mlflow.start_run(run_name=run_name) as run:
        yield run


def log_params(params: Dict[str, Any]) -> None:
    if not is_available():
        return
    import mlflow

    # MLflow rejects non-primitive values; stringify anything exotic (lists,
    # nested dicts from a hyperparameter grid) rather than crashing a
    # training run over a logging call.
    safe_params = {
        k: (v if isinstance(v, (str, int, float, bool)) else str(v))
        for k, v in params.items()
    }
    mlflow.log_params(safe_params)


def log_metrics(metrics: Dict[str, float]) -> None:
    if not is_available():
        return
    import mlflow

    numeric_metrics = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
    if numeric_metrics:
        mlflow.log_metrics(numeric_metrics)


def log_artifact(path: str) -> None:
    if not is_available():
        return
    import mlflow

    try:
        mlflow.log_artifact(path)
    except Exception:  # noqa: BLE001 - artifact logging is best-effort
        pass


def log_sklearn_model(model: Any, artifact_path: str) -> None:
    if not is_available():
        return
    import mlflow.sklearn

    try:
        mlflow.sklearn.log_model(model, artifact_path)
    except Exception:  # noqa: BLE001
        pass

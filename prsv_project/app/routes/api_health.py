from __future__ import annotations

from fastapi import APIRouter

from app.config import settings
from app.schemas import HealthStatus
from app.services.dataset_service import DatasetService
from ml.model_loader import load_model_artifacts

router = APIRouter(prefix="/api/health", tags=["health"])


@router.get("", response_model=HealthStatus)
def health_check() -> HealthStatus:
    dataset_service = DatasetService(settings)
    model_artifacts = load_model_artifacts(settings)

    return HealthStatus(
        status="ok",
        app_name=settings.app_name,
        app_version=settings.app_version,
        demo_dataset_available=dataset_service.demo_dataset_exists(),
        model_available=model_artifacts.model_available,
        kb_available=settings.kb_path.exists(),
    )
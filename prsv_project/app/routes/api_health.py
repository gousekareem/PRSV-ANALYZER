from __future__ import annotations

import uuid

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.config import settings
from app.schemas import HealthStatus
from app.services.dataset_service import DatasetService
from app.services.visitor_service import visitor_tracker
from ml.model_loader import load_model_artifacts

router = APIRouter(prefix="/api", tags=["health"])


@router.get("/health", response_model=HealthStatus)
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


@router.get("/visitor-stats")
def visitor_stats() -> dict:
    stats = visitor_tracker.get_stats()
    return {
        "total_visits": stats["total_visits"],
        "unique_visitors": stats["unique_visitors"],
        "online_now": stats["online_now"],
    }


@router.post("/track-visit")
async def track_visit(request: Request) -> JSONResponse:
    session_id = request.cookies.get("prsv_session_id") or uuid.uuid4().hex
    client_ip = request.client.host if request.client else "unknown"

    page = "/"
    try:
        if request.headers.get("content-type", "").startswith("application/json"):
            payload = await request.json()
            if isinstance(payload, dict):
                page = payload.get("page") or "/"
    except Exception:
        page = "/"

    stats = visitor_tracker.track_visit(session_id=session_id, ip_address=client_ip, page=page)

    response = JSONResponse({"success": True, **stats})
    response.set_cookie(
        key="prsv_session_id",
        value=session_id,
        max_age=30 * 24 * 60 * 60,
        httponly=True,
        samesite="lax",
    )
    return response

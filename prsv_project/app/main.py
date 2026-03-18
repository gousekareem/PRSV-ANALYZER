from __future__ import annotations

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.config import ROOT_DIR, settings
from app.routes.api_analysis import router as analysis_router
from app.routes.api_health import router as health_router
from app.routes.pages import router as pages_router
from app.utils.path_utils import ensure_dirs

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    debug=settings.debug,
)

ensure_dirs(
    [
        settings.upload_dir,
        settings.extracted_dir,
        settings.processed_dir,
        settings.output_dir,
        settings.temp_dir,
        settings.log_dir,
        settings.models_dir,
        settings.reports_dir,
    ]
)

app.mount("/static", StaticFiles(directory=str(ROOT_DIR / "app" / "static")), name="static")
app.mount("/outputs", StaticFiles(directory=str(settings.output_dir)), name="outputs")

app.include_router(pages_router)
app.include_router(analysis_router)
app.include_router(health_router)
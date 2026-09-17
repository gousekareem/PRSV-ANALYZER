from __future__ import annotations

import logging
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.config import ROOT_DIR, settings
from app.middleware import ApiKeyAndRateLimitMiddleware
from app.routes.api_analysis import router as analysis_router
from app.routes.api_chatbot import router as chatbot_router
from app.routes.api_health import router as health_router
from app.routes.pages import router as pages_router
from app.services.cleanup_service import run_cleanup
from app.utils.path_utils import ensure_dirs
from ml.model_loader import load_model_artifacts

_logger = logging.getLogger("prsv.startup")

# Scheduled cleanup runs once every this many seconds (24h). Kept as a plain
# constant rather than a setting since it's an operational interval, not
# something that needs per-deployment tuning the way retention *days* does.
_CLEANUP_INTERVAL_SECONDS = 24 * 60 * 60


def _warn_if_no_trained_model() -> None:
    """
    Loudly warn on startup if no trained SVM is present, so nobody mistakes
    heuristic-fallback predictions (a hand-tuned formula, not a classifier)
    for real model output.
    """
    artifacts = load_model_artifacts(settings)
    if artifacts.model_available:
        _logger.info("Trained SVM model loaded from %s - inference_mode will be 'trained_model'.", settings.models_dir)
        return

    banner = (
        "\n"
        + "=" * 78 + "\n"
        + "  WARNING: NO TRAINED MODEL FOUND\n"
        + f"  Expected: {settings.model_path}\n"
        + "  All predictions will use inference_mode='heuristic_fallback' - a\n"
        + "  hand-tuned weighted formula over 7 features, NOT a trained classifier.\n"
        + "  To train a real model:\n"
        + "    python scripts/auto_label_from_filenames.py\n"
        + "    python scripts/retrain_model.py\n"
        + "=" * 78 + "\n"
    )
    _logger.warning(banner)
    print(banner)


def _start_scheduled_cleanup() -> threading.Event:
    """
    Background daemon thread that runs the same retention cleanup as
    scripts/cleanup_outputs.py automatically, once every 24h, so data/outputs/
    doesn't grow unbounded even if nobody remembers to run the script or set
    up a scheduled task. Controlled by OUTPUT_RETENTION_DAYS in .env (0
    disables it). Returns a stop Event so it can be shut down cleanly.
    """
    stop_event = threading.Event()

    def _loop() -> None:
        while not stop_event.is_set():
            if settings.output_retention_days > 0:
                try:
                    result = run_cleanup(settings)
                    if result["deleted"]:
                        _logger.info(
                            "Scheduled cleanup: deleted %s run(s) older than %s day(s).",
                            result["deleted"],
                            settings.output_retention_days,
                        )
                except Exception:  # noqa: BLE001 - cleanup must never crash the app
                    _logger.exception("Scheduled cleanup failed.")
            stop_event.wait(_CLEANUP_INTERVAL_SECONDS)

    thread = threading.Thread(target=_loop, name="scheduled-cleanup", daemon=True)
    thread.start()
    return stop_event


def _warm_up_shap_explainer() -> None:
    """
    shap's first import/use in a process has a noticeable one-time cost
    (observed ~7s). Triggering it once at startup (rather than on the first
    real user request) keeps the app responsive from the first analysis
    onward. Best-effort: if this fails for any reason, the app still starts
    normally and SHAP will just lazy-load on first real use instead.
    """
    try:
        from ml.shap_explainer import explain_prediction

        dummy_vector = [0.5] * 12
        explain_prediction(dummy_vector, "Healthy", settings)
        _logger.info("SHAP explainer warmed up.")
    except Exception:  # noqa: BLE001
        _logger.info("SHAP warm-up skipped (will lazy-load on first use).")


@asynccontextmanager
async def lifespan(_: FastAPI):
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
    _warn_if_no_trained_model()
    _warm_up_shap_explainer()
    stop_cleanup = _start_scheduled_cleanup()
    yield
    stop_cleanup.set()


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    debug=settings.debug,
    lifespan=lifespan,
)

app.add_middleware(ApiKeyAndRateLimitMiddleware, settings=settings)

app.mount("/static", StaticFiles(directory=str(ROOT_DIR / "app" / "static")), name="static")
app.mount("/outputs", StaticFiles(directory=str(settings.output_dir)), name="outputs")

app.include_router(pages_router)
app.include_router(analysis_router)
app.include_router(health_router)
app.include_router(chatbot_router)
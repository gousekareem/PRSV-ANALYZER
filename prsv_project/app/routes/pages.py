from __future__ import annotations

from pathlib import Path
from typing import List

from fastapi import APIRouter, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from app.config import ROOT_DIR, settings
from app.services.analysis_service import AnalysisService
from app.services.batch_service import BatchService
from app.services.dataset_service import DatasetService
from app.services.run_manager import RunManager
from app.utils.file_utils import save_upload_file
from app.utils.json_utils import load_json
from app.utils.validation_utils import (
    is_allowed_extension,
    validate_image_readable,
    validate_non_empty_file,
)
from app.utils.zip_utils import safe_extract_zip
from ml.model_loader import load_model_artifacts

templates = Jinja2Templates(directory=str(ROOT_DIR / "app" / "templates"))

router = APIRouter(tags=["pages"])


def _list_runs(limit: int | None = None) -> list[dict]:
    runs = []
    if settings.output_dir.exists():
        for run_dir in sorted(settings.output_dir.iterdir(), reverse=True):
            if not run_dir.is_dir() or not run_dir.name.startswith("run_"):
                continue

            summary_path = run_dir / "batch_summary.json"
            item = {"run_id": run_dir.name}
            if summary_path.exists():
                try:
                    summary = load_json(summary_path)
                    item.update(summary)
                except Exception:
                    item["processed_images"] = 0
                    item["failed_images"] = 0
            runs.append(item)

    if limit is not None:
        return runs[:limit]
    return runs


def _filter_runs(
    runs: list[dict],
    search: str | None = None,
    status_filter: str | None = None,
    severity_filter: str | None = None,
    sort_by: str = "newest",
) -> list[dict]:
    filtered = runs

    if search:
        search_lower = search.strip().lower()
        filtered = [run for run in filtered if search_lower in str(run.get("run_id", "")).lower()]

    if status_filter == "healthy":
        filtered = [run for run in filtered if (run.get("healthy_count", 0) > 0 and run.get("diseased_count", 0) == 0)]
    elif status_filter == "diseased":
        filtered = [run for run in filtered if run.get("diseased_count", 0) > 0]
    elif status_filter == "mixed":
        filtered = [
            run for run in filtered
            if run.get("healthy_count", 0) > 0 and run.get("diseased_count", 0) > 0
        ]

    if severity_filter:
        filtered = [
            run for run in filtered
            if severity_filter in (run.get("severity_distribution") or {})
        ]

    if sort_by == "oldest":
        filtered = list(reversed(filtered))
    else:
        filtered = filtered

    return filtered


def _build_services() -> tuple[RunManager, AnalysisService, BatchService]:
    run_manager = RunManager(settings)
    analysis_service = AnalysisService(settings, run_manager)
    batch_service = BatchService(settings, run_manager, analysis_service)
    return run_manager, analysis_service, batch_service


@router.get("/", response_class=HTMLResponse)
def home_page(request: Request) -> HTMLResponse:
    dataset_service = DatasetService(settings)
    demo_count = len(dataset_service.list_demo_images()) if dataset_service.demo_dataset_exists() else 0
    recent_runs = _list_runs(limit=5)

    return templates.TemplateResponse(
        request,
        "home.html",
        {
            "app_name": settings.app_name,
            "demo_count": demo_count,
            "recent_runs": recent_runs,
        },
    )


@router.get("/analyze", response_class=HTMLResponse)
def analyze_page(
    request: Request,
    message: str | None = Query(default=None),
    level: str | None = Query(default=None),
) -> HTMLResponse:
    dataset_service = DatasetService(settings)

    return templates.TemplateResponse(
        request,
        "analyze.html",
        {
            "demo_dataset_exists": dataset_service.demo_dataset_exists(),
            "demo_count": len(dataset_service.list_demo_images()) if dataset_service.demo_dataset_exists() else 0,
            "message": message,
            "level": level,
        },
    )


@router.get("/status", response_class=HTMLResponse)
def status_page(request: Request) -> HTMLResponse:
    dataset_service = DatasetService(settings)
    model_artifacts = load_model_artifacts(settings)

    writable_checks = {
        "uploads_writable": settings.upload_dir.exists(),
        "outputs_writable": settings.output_dir.exists(),
        "temp_writable": settings.temp_dir.exists(),
    }

    context = {
        "app_name": settings.app_name,
        "app_version": settings.app_version,
        "demo_dataset_available": dataset_service.demo_dataset_exists(),
        "demo_count": len(dataset_service.list_demo_images()) if dataset_service.demo_dataset_exists() else 0,
        "model_available": model_artifacts.model_available,
        "kb_available": settings.kb_path.exists(),
        "writable_checks": writable_checks,
        "recent_runs": _list_runs(limit=5),
    }

    return templates.TemplateResponse(request, "status.html", context)


@router.get("/methodology", response_class=HTMLResponse)
def methodology_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "methodology.html",
        {
            "app_name": settings.app_name,
        },
    )


@router.get("/results", response_class=HTMLResponse)
def results_page(
    request: Request,
    search: str | None = Query(default=None),
    status_filter: str | None = Query(default=None),
    severity_filter: str | None = Query(default=None),
    sort_by: str = Query(default="newest"),
) -> HTMLResponse:
    runs = _list_runs()
    filtered_runs = _filter_runs(
        runs=runs,
        search=search,
        status_filter=status_filter,
        severity_filter=severity_filter,
        sort_by=sort_by,
    )

    return templates.TemplateResponse(
        request,
        "result.html",
        {
            "runs": filtered_runs,
            "search": search or "",
            "status_filter": status_filter or "",
            "severity_filter": severity_filter or "",
            "sort_by": sort_by,
        },
    )


@router.get("/batch-results", response_class=HTMLResponse)
def batch_results_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "batch_result.html",
        {
            "runs": _list_runs(),
        },
    )


@router.get("/run/{run_id}", response_class=HTMLResponse)
def run_detail_page(request: Request, run_id: str) -> HTMLResponse:
    summary_path = settings.output_dir / run_id / "batch_summary.json"
    if not summary_path.exists():
        raise HTTPException(status_code=404, detail="Run not found.")

    summary = load_json(summary_path)

    return templates.TemplateResponse(
        request,
        "run_detail.html",
        {
            "run_id": run_id,
            "summary": summary,
        },
    )


@router.get("/run/{run_id}/image/{image_id}", response_class=HTMLResponse)
def image_detail_page(request: Request, run_id: str, image_id: str) -> HTMLResponse:
    image_dir = settings.output_dir / run_id / "images" / image_id
    if not image_dir.exists():
        raise HTTPException(status_code=404, detail="Image result not found.")

    def maybe_load(name: str) -> dict:
        path = image_dir / name
        return load_json(path) if path.exists() else {}

    image_payload = {
        "features": maybe_load("features.json"),
        "prediction": maybe_load("prediction.json"),
        "severity": maybe_load("severity.json"),
        "rag": maybe_load("rag.json"),
        "explanation": maybe_load("explanation.json"),
        "paths": {},
    }

    for item in image_dir.iterdir():
        if item.is_file():
            image_payload["paths"][item.name] = f"/outputs/{run_id}/images/{image_id}/{item.name}"

    return templates.TemplateResponse(
        request,
        "image_detail.html",
        {
            "run_id": run_id,
            "image_id": image_id,
            "image_payload": image_payload,
        },
    )


@router.post("/analyze/single")
async def analyze_single_form(file: UploadFile = File(...)) -> RedirectResponse:
    _, analysis_service, _ = _build_services()

    if not is_allowed_extension(file.filename or "", settings.allowed_extensions):
        return RedirectResponse(
            url="/analyze?message=Unsupported%20file%20type&level=error",
            status_code=303,
        )

    saved_path = await save_upload_file(file, settings.upload_dir)

    if not validate_non_empty_file(saved_path):
        return RedirectResponse(
            url="/analyze?message=Uploaded%20file%20is%20empty&level=error",
            status_code=303,
        )

    if not validate_image_readable(saved_path):
        return RedirectResponse(
            url="/analyze?message=Uploaded%20image%20is%20unreadable%20or%20corrupt&level=error",
            status_code=303,
        )

    result = analysis_service.analyze_single_image(saved_path)

    run_id = Path(result.output_paths["features_json"]).parents[1].name
    return RedirectResponse(
        url=f"/run/{run_id}/image/{result.image_id}",
        status_code=303,
    )


@router.post("/analyze/multiple")
async def analyze_multiple_form(files: List[UploadFile] = File(...)) -> RedirectResponse:
    _, _, batch_service = _build_services()

    if not files:
        return RedirectResponse(
            url="/analyze?message=No%20files%20were%20uploaded&level=error",
            status_code=303,
        )

    saved_paths: List[Path] = []

    for file in files:
        if not is_allowed_extension(file.filename or "", settings.allowed_extensions):
            continue

        saved_path = await save_upload_file(file, settings.upload_dir)
        if validate_non_empty_file(saved_path) and validate_image_readable(saved_path):
            saved_paths.append(saved_path)

    if not saved_paths:
        return RedirectResponse(
            url="/analyze?message=No%20valid%20readable%20images%20found&level=error",
            status_code=303,
        )

    batch_result = batch_service.analyze_images(saved_paths)
    return RedirectResponse(
        url=f"/run/{batch_result.run_id}",
        status_code=303,
    )


@router.post("/analyze/zip")
async def analyze_zip_form(file: UploadFile = File(...)) -> RedirectResponse:
    _, _, batch_service = _build_services()

    filename = file.filename or ""
    if Path(filename).suffix.lower() != ".zip":
        return RedirectResponse(
            url="/analyze?message=Only%20ZIP%20files%20are%20supported&level=error",
            status_code=303,
        )

    saved_zip_path = await save_upload_file(file, settings.temp_dir)

    if not validate_non_empty_file(saved_zip_path):
        return RedirectResponse(
            url="/analyze?message=Uploaded%20ZIP%20file%20is%20empty&level=error",
            status_code=303,
        )

    extraction_dir = settings.extracted_dir / saved_zip_path.stem

    try:
        extracted_files = safe_extract_zip(
            zip_path=saved_zip_path,
            extract_dir=extraction_dir,
            allowed_extensions=settings.allowed_extensions,
        )
    except ValueError as exc:
        return RedirectResponse(
            url=f"/analyze?message={str(exc).replace(' ', '%20')}&level=error",
            status_code=303,
        )

    valid_images: List[Path] = []
    for path in extracted_files:
        if validate_non_empty_file(path) and validate_image_readable(path):
            valid_images.append(path)

    if not valid_images:
        return RedirectResponse(
            url="/analyze?message=No%20valid%20readable%20images%20were%20found%20in%20the%20ZIP%20archive&level=error",
            status_code=303,
        )

    batch_result = batch_service.analyze_images(valid_images)
    return RedirectResponse(
        url=f"/run/{batch_result.run_id}",
        status_code=303,
    )


@router.post("/analyze/demo")
def analyze_demo_form(limit: int = Form(default=10)) -> RedirectResponse:
    _, _, batch_service = _build_services()
    dataset_service = DatasetService(settings)

    if not dataset_service.demo_dataset_exists():
        return RedirectResponse(
            url="/analyze?message=Demo%20dataset%20path%20not%20found&level=error",
            status_code=303,
        )

    image_paths = dataset_service.get_demo_sample(limit=limit)
    if not image_paths:
        return RedirectResponse(
            url="/analyze?message=No%20demo%20dataset%20images%20were%20found&level=error",
            status_code=303,
        )

    batch_result = batch_service.analyze_images(image_paths)
    return RedirectResponse(
        url=f"/run/{batch_result.run_id}",
        status_code=303,
    )
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse

from app.config import settings
from app.schemas import BatchResult, ImageResult
from app.services.analysis_service import AnalysisService
from app.services.batch_service import BatchService
from app.services.dataset_service import DatasetService
from app.services.export_service import ExportService
from app.services.job_queue import BatchJobQueue
from app.services.report_service import generate_image_report_pdf, generate_run_report_pdf
from app.services.run_manager import RunManager
from app.services.run_store import RunStore
from app.utils.file_utils import save_upload_file
from app.utils.json_utils import load_json
from app.utils.path_utils import is_within_directory
from app.utils.validation_utils import (
    is_allowed_extension,
    validate_image_readable,
    validate_non_empty_file,
)
from app.utils.zip_utils import safe_extract_zip

router = APIRouter(prefix="/api/analysis", tags=["analysis"])

# run_id looks like: run_2026_07_02_14_30_05_a1b2c3 (see app.utils.id_utils.generate_run_id)
# image_id looks like: img_a1b2c3d4e5 (see app.utils.id_utils.generate_image_id)
_RUN_ID_PATTERN = re.compile(r"^run_\d{4}(?:_\d{2}){5}_[0-9a-f]{6}$")
_IMAGE_ID_PATTERN = re.compile(r"^img_[0-9a-f]{6,16}$")


def _resolve_run_dir(run_id: str) -> Path:
    """
    Validate run_id against a strict whitelist pattern and confirm the
    resolved path stays inside output_dir before it is ever used in a
    filesystem lookup. Rejects anything with '..', separators, or a shape
    that doesn't match what RunManager actually generates.
    """
    if not _RUN_ID_PATTERN.match(run_id):
        raise HTTPException(status_code=400, detail="Invalid run_id format.")

    run_dir = (settings.output_dir / run_id).resolve()
    if not is_within_directory(settings.output_dir, run_dir):
        raise HTTPException(status_code=400, detail="Invalid run_id.")

    return run_dir


def _resolve_image_dir(run_id: str, image_id: str) -> Path:
    run_dir = _resolve_run_dir(run_id)

    if not _IMAGE_ID_PATTERN.match(image_id):
        raise HTTPException(status_code=400, detail="Invalid image_id format.")

    image_dir = (run_dir / "images" / image_id).resolve()
    if not is_within_directory(run_dir, image_dir):
        raise HTTPException(status_code=400, detail="Invalid image_id.")

    return image_dir


def build_services() -> tuple[RunManager, AnalysisService, BatchService]:
    run_manager = RunManager(settings)
    analysis_service = AnalysisService(settings, run_manager)
    batch_service = BatchService(settings, run_manager, analysis_service)
    return run_manager, analysis_service, batch_service


async def _collect_valid_uploads(files: List[UploadFile]) -> List[Path]:
    if not files:
        raise HTTPException(status_code=400, detail="No files were uploaded.")

    saved_paths: List[Path] = []
    for file in files:
        if not is_allowed_extension(file.filename or "", settings.allowed_extensions):
            continue

        saved_path = await save_upload_file(file, settings.upload_dir)
        if validate_non_empty_file(saved_path) and validate_image_readable(saved_path):
            saved_paths.append(saved_path)

    if not saved_paths:
        raise HTTPException(status_code=400, detail="No valid readable images found in upload.")

    return saved_paths


async def _collect_valid_zip_images(file: UploadFile) -> List[Path]:
    filename = file.filename or ""
    if Path(filename).suffix.lower() != ".zip":
        raise HTTPException(status_code=400, detail="Only ZIP files are supported for this endpoint.")

    saved_zip_path = await save_upload_file(file, settings.temp_dir)

    if not validate_non_empty_file(saved_zip_path):
        raise HTTPException(status_code=400, detail="Uploaded ZIP file is empty.")

    extraction_dir = settings.extracted_dir / saved_zip_path.stem
    try:
        extracted_files = safe_extract_zip(
            zip_path=saved_zip_path,
            extract_dir=extraction_dir,
            allowed_extensions=settings.allowed_extensions,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    valid_images: List[Path] = []
    for path in extracted_files:
        if validate_non_empty_file(path) and validate_image_readable(path):
            valid_images.append(path)

    if not valid_images:
        raise HTTPException(status_code=400, detail="No valid readable images were found in the ZIP archive.")

    return valid_images


def _collect_demo_images(limit: int) -> List[Path]:
    dataset_service = DatasetService(settings)

    if not dataset_service.demo_dataset_exists():
        raise HTTPException(status_code=404, detail="Demo dataset path not found.")

    image_paths = dataset_service.get_demo_sample(limit=limit)
    if not image_paths:
        raise HTTPException(status_code=404, detail="No demo dataset images were found.")

    return image_paths


@router.post("/single", response_model=ImageResult)
async def analyze_single(file: UploadFile = File(...)) -> ImageResult:
    _, _, batch_service = build_services()

    if not is_allowed_extension(file.filename or "", settings.allowed_extensions):
        raise HTTPException(status_code=400, detail="Unsupported file type.")

    saved_path = await save_upload_file(file, settings.upload_dir)

    if not validate_non_empty_file(saved_path):
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    if not validate_image_readable(saved_path):
        raise HTTPException(status_code=400, detail="Uploaded image is unreadable or corrupt.")

    # Goes through batch_service (even for a single image) so batch_summary.json,
    # the CSV export, and charts always exist for this run - keeping /run/{run_id}
    # and /run/{run_id}/download working uniformly regardless of how many images
    # were submitted.
    batch_result = batch_service.analyze_images([saved_path])
    if not batch_result.results:
        raise HTTPException(status_code=422, detail="Image could not be processed.")

    return batch_result.results[0]


@router.post("/multiple", response_model=BatchResult)
async def analyze_multiple(files: List[UploadFile] = File(...)) -> BatchResult:
    _, _, batch_service = build_services()
    saved_paths = await _collect_valid_uploads(files)
    return batch_service.analyze_images(saved_paths)


@router.post("/multiple-async")
async def analyze_multiple_async(files: List[UploadFile] = File(...)) -> dict:
    """
    Same as /multiple, but returns immediately with a job_id instead of
    blocking until every image is processed. Poll GET /job/{job_id} for
    progress. Recommended for large batches so the request thread isn't
    tied up for the whole run.
    """
    _, _, batch_service = build_services()
    saved_paths = await _collect_valid_uploads(files)
    job_id = BatchJobQueue.get_instance(settings).submit(batch_service, saved_paths)
    return {"job_id": job_id, "status": "queued", "total_images": len(saved_paths)}


@router.post("/zip", response_model=BatchResult)
async def analyze_zip(file: UploadFile = File(...)) -> BatchResult:
    _, _, batch_service = build_services()
    valid_images = await _collect_valid_zip_images(file)
    return batch_service.analyze_images(valid_images)


@router.post("/zip-async")
async def analyze_zip_async(file: UploadFile = File(...)) -> dict:
    _, _, batch_service = build_services()
    valid_images = await _collect_valid_zip_images(file)
    job_id = BatchJobQueue.get_instance(settings).submit(batch_service, valid_images)
    return {"job_id": job_id, "status": "queued", "total_images": len(valid_images)}


@router.post("/demo", response_model=BatchResult)
def analyze_demo_dataset(
    limit: int = Query(default=10, ge=1, le=200),
) -> BatchResult:
    _, _, batch_service = build_services()
    image_paths = _collect_demo_images(limit)
    return batch_service.analyze_images(image_paths)


@router.post("/demo-async")
def analyze_demo_dataset_async(
    limit: int = Query(default=10, ge=1, le=200),
) -> dict:
    _, _, batch_service = build_services()
    image_paths = _collect_demo_images(limit)
    job_id = BatchJobQueue.get_instance(settings).submit(batch_service, image_paths)
    return {"job_id": job_id, "status": "queued", "total_images": len(image_paths)}


@router.get("/job/{job_id}")
def get_job_status(job_id: str) -> dict:
    if not re.match(r"^job_[0-9a-f]{12}$", job_id):
        raise HTTPException(status_code=400, detail="Invalid job_id format.")

    job = BatchJobQueue.get_instance(settings).get_status(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return job


@router.get("/runs")
def list_runs() -> dict:
    runs = []
    if settings.output_dir.exists():
        for run_dir in sorted(settings.output_dir.iterdir(), reverse=True):
            if not run_dir.is_dir() or not run_dir.name.startswith("run_"):
                continue

            summary_path = run_dir / "batch_summary.json"
            run_info = {
                "run_id": run_dir.name,
                "has_summary": summary_path.exists(),
            }

            if summary_path.exists():
                try:
                    summary = load_json(summary_path)
                    run_info.update(
                        {
                            "processed_images": summary.get("processed_images", 0),
                            "failed_images": summary.get("failed_images", 0),
                            "healthy_count": summary.get("healthy_count", 0),
                            "diseased_count": summary.get("diseased_count", 0),
                        }
                    )
                except Exception:
                    pass

            runs.append(run_info)

    return {"runs": runs}


@router.get("/runs/search")
def search_runs(
    q: Optional[str] = Query(default=None, description="Search run_id or filenames"),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=200),
    sort_by: str = Query(default="created_at"),
    sort_dir: str = Query(default="desc"),
) -> dict:
    """
    SQLite-backed search/filter/sort/pagination over run metadata. Faster
    than /runs (which scans the filesystem) once there are many runs, and
    supports search - this is what the Runs/Batch pages' frontend uses.
    """
    run_store = RunStore(settings)
    return run_store.list_runs(query=q, page=page, page_size=page_size, sort_by=sort_by, sort_dir=sort_dir)


@router.get("/run/{run_id}")
def get_run_summary(run_id: str) -> dict:
    run_dir = _resolve_run_dir(run_id)
    summary_path = run_dir / "batch_summary.json"

    if not summary_path.exists():
        raise HTTPException(status_code=404, detail="Run summary not found.")

    return load_json(summary_path)


@router.get("/run/{run_id}/image/{image_id}")
def get_image_details(run_id: str, image_id: str) -> dict:
    image_dir = _resolve_image_dir(run_id, image_id)
    if not image_dir.exists():
        raise HTTPException(status_code=404, detail="Image result directory not found.")

    files = {
        "features": image_dir / "features.json",
        "prediction": image_dir / "prediction.json",
        "severity": image_dir / "severity.json",
        "rag": image_dir / "rag.json",
        "explanation": image_dir / "explanation.json",
    }

    payload = {
        "run_id": run_id,
        "image_id": image_id,
        "paths": {},
        "json": {},
    }

    for key, path in files.items():
        if path.exists():
            payload["json"][key] = load_json(path)

    for item in image_dir.iterdir():
        if item.is_file():
            payload["paths"][item.name] = f"/outputs/{run_id}/images/{image_id}/{item.name}"

    return payload


@router.get("/run/{run_id}/download")
def download_run_bundle(run_id: str) -> FileResponse:
    run_dir = _resolve_run_dir(run_id)
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail="Run directory not found.")

    exporter = ExportService()
    archive_path = exporter.create_run_bundle(run_dir)

    return FileResponse(
        path=str(archive_path),
        filename=archive_path.name,
        media_type="application/zip",
    )


@router.get("/run/{run_id}/image/{image_id}/report.pdf")
def download_image_report(run_id: str, image_id: str) -> FileResponse:
    image_dir = _resolve_image_dir(run_id, image_id)
    if not image_dir.exists():
        raise HTTPException(status_code=404, detail="Image result directory not found.")

    pdf_path = generate_image_report_pdf(settings, run_id, image_id, image_dir)
    return FileResponse(
        path=str(pdf_path),
        filename=f"prsv_report_{image_id}.pdf",
        media_type="application/pdf",
    )


@router.get("/run/{run_id}/report.pdf")
def download_run_report(run_id: str) -> FileResponse:
    run_dir = _resolve_run_dir(run_id)
    summary_path = run_dir / "batch_summary.json"
    if not summary_path.exists():
        raise HTTPException(status_code=404, detail="Run summary not found.")

    pdf_path = generate_run_report_pdf(settings, run_id, run_dir)
    return FileResponse(
        path=str(pdf_path),
        filename=f"prsv_run_report_{run_id}.pdf",
        media_type="application/pdf",
    )
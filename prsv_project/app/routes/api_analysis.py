from __future__ import annotations

from pathlib import Path
from typing import List

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse

from app.config import settings
from app.schemas import BatchResult, ImageResult
from app.services.analysis_service import AnalysisService
from app.services.batch_service import BatchService
from app.services.dataset_service import DatasetService
from app.services.export_service import ExportService
from app.services.run_manager import RunManager
from app.utils.file_utils import save_upload_file
from app.utils.json_utils import load_json
from app.utils.validation_utils import (
    is_allowed_extension,
    validate_image_readable,
    validate_non_empty_file,
)
from app.utils.zip_utils import safe_extract_zip

router = APIRouter(prefix="/api/analysis", tags=["analysis"])


def build_services() -> tuple[RunManager, AnalysisService, BatchService]:
    run_manager = RunManager(settings)
    analysis_service = AnalysisService(settings, run_manager)
    batch_service = BatchService(settings, run_manager, analysis_service)
    return run_manager, analysis_service, batch_service


@router.post("/single", response_model=ImageResult)
async def analyze_single(file: UploadFile = File(...)) -> ImageResult:
    _, analysis_service, _ = build_services()

    if not is_allowed_extension(file.filename or "", settings.allowed_extensions):
        raise HTTPException(status_code=400, detail="Unsupported file type.")

    saved_path = await save_upload_file(file, settings.upload_dir)

    if not validate_non_empty_file(saved_path):
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    if not validate_image_readable(saved_path):
        raise HTTPException(status_code=400, detail="Uploaded image is unreadable or corrupt.")

    return analysis_service.analyze_single_image(saved_path)


@router.post("/multiple", response_model=BatchResult)
async def analyze_multiple(files: List[UploadFile] = File(...)) -> BatchResult:
    _, _, batch_service = build_services()

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

    return batch_service.analyze_images(saved_paths)


@router.post("/zip", response_model=BatchResult)
async def analyze_zip(file: UploadFile = File(...)) -> BatchResult:
    _, _, batch_service = build_services()

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

    return batch_service.analyze_images(valid_images)


@router.post("/demo", response_model=BatchResult)
def analyze_demo_dataset(
    limit: int = Query(default=10, ge=1, le=200),
) -> BatchResult:
    _, _, batch_service = build_services()
    dataset_service = DatasetService(settings)

    if not dataset_service.demo_dataset_exists():
        raise HTTPException(status_code=404, detail="Demo dataset path not found.")

    image_paths = dataset_service.get_demo_sample(limit=limit)
    if not image_paths:
        raise HTTPException(status_code=404, detail="No demo dataset images were found.")

    return batch_service.analyze_images(image_paths)


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
                "path": str(run_dir),
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


@router.get("/run/{run_id}")
def get_run_summary(run_id: str) -> dict:
    run_dir = settings.output_dir / run_id
    summary_path = run_dir / "batch_summary.json"

    if not summary_path.exists():
        raise HTTPException(status_code=404, detail="Run summary not found.")

    return load_json(summary_path)


@router.get("/run/{run_id}/image/{image_id}")
def get_image_details(run_id: str, image_id: str) -> dict:
    image_dir = settings.output_dir / run_id / "images" / image_id
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
    run_dir = settings.output_dir / run_id
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail="Run directory not found.")

    exporter = ExportService()
    archive_path = exporter.create_run_bundle(run_dir)

    return FileResponse(
        path=str(archive_path),
        filename=archive_path.name,
        media_type="application/zip",
    )
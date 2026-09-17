from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from app.config import Settings
from app.schemas import BatchResult, ImageResult
from app.services.analysis_service import AnalysisService
from app.services.export_service import ExportService
from app.services.run_manager import RunManager
from app.services.run_store import RunStore
from app.utils.json_utils import save_json
from app.utils.logging_utils import get_logger


class BatchService:
    """
    Handles multi-image processing and batch summary creation.
    """

    def __init__(self, settings: Settings, run_manager: RunManager, analysis_service: AnalysisService) -> None:
        self.settings = settings
        self.run_manager = run_manager
        self.analysis_service = analysis_service
        self.export_service = ExportService()
        self.run_store = RunStore(settings)

    def analyze_images(self, image_paths: List[Path], progress_callback=None) -> BatchResult:
        run_context = self.run_manager.create_run()
        logger = get_logger("batch_service", run_context.log_file)

        results: List[ImageResult] = []
        failures: List[Dict[str, Any]] = []

        logger.info("Batch analysis started. Total input images: %s", len(image_paths))

        for idx, image_path in enumerate(image_paths):
            try:
                result = self.analysis_service.analyze_single_image(
                    image_path=image_path,
                    run_context=run_context,
                )
                results.append(result)
            except Exception as exc:
                logger.exception("Failed to process image: %s", image_path)
                failures.append(
                    {
                        "filename": image_path.name,
                        "error": str(exc),
                    }
                )
            finally:
                if progress_callback is not None:
                    try:
                        progress_callback(idx + 1, len(image_paths))
                    except Exception:  # noqa: BLE001 - progress reporting must never break the batch
                        pass

        processed_images = len(results)
        failed_images = len(failures)

        healthy_count = sum(1 for item in results if item.prediction.lower() == "healthy")
        diseased_count = sum(1 for item in results if item.prediction.lower() != "healthy")

        average_confidence = (
            sum(item.confidence for item in results) / processed_images if processed_images else 0.0
        )
        average_infection_percentage = (
            sum(item.infection_percentage for item in results) / processed_images if processed_images else 0.0
        )

        severity_distribution: Dict[str, int] = {}
        for item in results:
            severity_distribution[item.severity_label] = severity_distribution.get(item.severity_label, 0) + 1

        batch_result = BatchResult(
            run_id=run_context.run_id,
            total_images=len(image_paths),
            processed_images=processed_images,
            failed_images=failed_images,
            healthy_count=healthy_count,
            diseased_count=diseased_count,
            average_confidence=round(average_confidence, 4),
            average_infection_percentage=round(average_infection_percentage, 4),
            severity_distribution=severity_distribution,
            results=results,
            failures=failures,
        )

        save_json(run_context.run_dir / "batch_summary.json", batch_result.model_dump())
        self.export_service.export_batch_csv(run_context.run_dir, batch_result)
        self.export_service.generate_batch_charts(run_context.run_dir, batch_result)

        # created_at is parsed straight from the run_id (run_YYYY_MM_DD_HH_MM_SS_xxxxxx)
        # so it always matches the folder's actual timestamp, not "now" (relevant if
        # this ever runs from a queued background job with some processing delay).
        try:
            ts_part = "_".join(run_context.run_id.split("_")[1:7])
            created_at = datetime.strptime(ts_part, "%Y_%m_%d_%H_%M_%S").isoformat()
        except ValueError:
            created_at = datetime.now().isoformat()

        self.run_store.upsert_run(
            run_id=run_context.run_id,
            created_at=created_at,
            total_images=batch_result.total_images,
            processed_images=batch_result.processed_images,
            failed_images=batch_result.failed_images,
            healthy_count=batch_result.healthy_count,
            diseased_count=batch_result.diseased_count,
            average_confidence=batch_result.average_confidence,
            average_infection_percentage=batch_result.average_infection_percentage,
            filenames=[item.filename for item in results],
        )

        logger.info(
            "Batch analysis completed | run_id=%s | success=%s | failed=%s",
            run_context.run_id,
            processed_images,
            failed_images,
        )

        return batch_result
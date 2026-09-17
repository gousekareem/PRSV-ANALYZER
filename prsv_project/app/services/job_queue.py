from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import List
from uuid import uuid4

from app.config import Settings
from app.services.batch_service import BatchService
from app.services.run_store import RunStore


class BatchJobQueue:
    """
    Lightweight in-process background job queue for batch/ZIP/demo analysis.

    Deliberately not Celery/Redis: this app is designed to run as a single
    process on one machine (a Windows laptop, per this project's deployment
    target), so a full distributed task queue would add operational
    complexity (a broker to run, more moving parts to keep alive) without a
    matching benefit. A single-worker ThreadPoolExecutor gives the same UX
    win - the HTTP request returns immediately with a job_id, the frontend
    polls for progress, and large batches no longer block the request thread
    - without needing anything else installed or running.

    If this ever needs to scale beyond one machine, swapping this class for
    a Celery-backed implementation behind the same submit/status interface
    is a contained change - routes and frontend polling stay the same.
    """

    _instance: "BatchJobQueue | None" = None
    _lock = threading.Lock()

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.run_store = RunStore(settings)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="batch-job")

    @classmethod
    def get_instance(cls, settings: Settings) -> "BatchJobQueue":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(settings)
            return cls._instance

    def submit(self, batch_service: BatchService, image_paths: List[Path]) -> str:
        job_id = f"job_{uuid4().hex[:12]}"
        now = datetime.now().isoformat()
        self.run_store.create_job(job_id=job_id, created_at=now, total_images=len(image_paths))

        def _progress(done: int, total: int) -> None:
            self.run_store.update_job(
                job_id=job_id,
                status="processing",
                updated_at=datetime.now().isoformat(),
                processed_images=done,
            )

        def _run() -> None:
            self.run_store.update_job(job_id=job_id, status="processing", updated_at=datetime.now().isoformat())
            try:
                batch_result = batch_service.analyze_images(image_paths, progress_callback=_progress)
                self.run_store.update_job(
                    job_id=job_id,
                    status="done",
                    updated_at=datetime.now().isoformat(),
                    processed_images=batch_result.processed_images,
                    run_id=batch_result.run_id,
                )
            except Exception as exc:  # noqa: BLE001 - surfaced to the polling client, not swallowed
                self.run_store.update_job(
                    job_id=job_id,
                    status="error",
                    updated_at=datetime.now().isoformat(),
                    error=str(exc),
                )

        self._executor.submit(_run)
        return job_id

    def get_status(self, job_id: str) -> dict | None:
        return self.run_store.get_job(job_id)

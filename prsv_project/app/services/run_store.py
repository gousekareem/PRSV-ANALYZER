from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from app.config import Settings

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    total_images INTEGER NOT NULL DEFAULT 0,
    processed_images INTEGER NOT NULL DEFAULT 0,
    failed_images INTEGER NOT NULL DEFAULT 0,
    healthy_count INTEGER NOT NULL DEFAULT 0,
    diseased_count INTEGER NOT NULL DEFAULT 0,
    average_confidence REAL NOT NULL DEFAULT 0,
    average_infection_percentage REAL NOT NULL DEFAULT 0,
    source TEXT NOT NULL DEFAULT 'batch',
    filenames TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_runs_created_at ON runs (created_at DESC);

CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    status TEXT NOT NULL DEFAULT 'queued',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    run_id TEXT,
    total_images INTEGER NOT NULL DEFAULT 0,
    processed_images INTEGER NOT NULL DEFAULT 0,
    error TEXT
);
"""


class RunStore:
    """
    Lightweight SQLite index over run metadata. This is additive, not a
    replacement for the JSON files under data/outputs/<run_id>/ - those
    remain the source of truth for a run's full detail. This index exists
    purely so the Runs/Batch pages can search, filter, sort, and paginate
    without scanning the filesystem on every request.
    """

    def __init__(self, settings: Settings) -> None:
        self.db_path: Path = settings.data_dir / "prsv_index.sqlite3"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.db_path), timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def upsert_run(
        self,
        run_id: str,
        created_at: str,
        total_images: int,
        processed_images: int,
        failed_images: int,
        healthy_count: int,
        diseased_count: int,
        average_confidence: float,
        average_infection_percentage: float,
        filenames: List[str],
        source: str = "batch",
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO runs (
                    run_id, created_at, total_images, processed_images, failed_images,
                    healthy_count, diseased_count, average_confidence,
                    average_infection_percentage, source, filenames
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    total_images=excluded.total_images,
                    processed_images=excluded.processed_images,
                    failed_images=excluded.failed_images,
                    healthy_count=excluded.healthy_count,
                    diseased_count=excluded.diseased_count,
                    average_confidence=excluded.average_confidence,
                    average_infection_percentage=excluded.average_infection_percentage,
                    filenames=excluded.filenames
                """,
                (
                    run_id, created_at, total_images, processed_images, failed_images,
                    healthy_count, diseased_count, average_confidence,
                    average_infection_percentage, source, " ".join(filenames),
                ),
            )

    def delete_run(self, run_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))

    def list_runs(
        self,
        query: Optional[str] = None,
        page: int = 1,
        page_size: int = 20,
        sort_by: str = "created_at",
        sort_dir: str = "desc",
    ) -> Dict[str, Any]:
        page = max(1, page)
        page_size = max(1, min(page_size, 200))
        offset = (page - 1) * page_size

        allowed_sort_columns = {
            "created_at", "total_images", "processed_images", "healthy_count",
            "diseased_count", "average_confidence", "average_infection_percentage",
        }
        sort_by = sort_by if sort_by in allowed_sort_columns else "created_at"
        sort_dir = "ASC" if sort_dir.lower() == "asc" else "DESC"

        where_clause = ""
        params: List[Any] = []
        if query:
            where_clause = "WHERE run_id LIKE ? OR filenames LIKE ?"
            like_query = f"%{query}%"
            params.extend([like_query, like_query])

        with self._connect() as conn:
            total_row = conn.execute(f"SELECT COUNT(*) as c FROM runs {where_clause}", params).fetchone()
            total = total_row["c"] if total_row else 0

            rows = conn.execute(
                f"""
                SELECT * FROM runs {where_clause}
                ORDER BY {sort_by} {sort_dir}
                LIMIT ? OFFSET ?
                """,
                [*params, page_size, offset],
            ).fetchall()

        return {
            "items": [dict(row) for row in rows],
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": max(1, (total + page_size - 1) // page_size),
        }

    # --- Background job tracking (for the async batch queue) ---

    def create_job(self, job_id: str, created_at: str, total_images: int) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO jobs (job_id, status, created_at, updated_at, total_images)
                VALUES (?, 'queued', ?, ?, ?)
                """,
                (job_id, created_at, created_at, total_images),
            )

    def update_job(
        self,
        job_id: str,
        status: str,
        updated_at: str,
        processed_images: Optional[int] = None,
        run_id: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        with self._connect() as conn:
            fields = ["status = ?", "updated_at = ?"]
            params: List[Any] = [status, updated_at]
            if processed_images is not None:
                fields.append("processed_images = ?")
                params.append(processed_images)
            if run_id is not None:
                fields.append("run_id = ?")
                params.append(run_id)
            if error is not None:
                fields.append("error = ?")
                params.append(error)
            params.append(job_id)
            conn.execute(f"UPDATE jobs SET {', '.join(fields)} WHERE job_id = ?", params)

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        return dict(row) if row else None

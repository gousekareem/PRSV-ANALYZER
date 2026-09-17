from __future__ import annotations

import shutil
from datetime import datetime, timedelta
from typing import Dict

from app.config import Settings
from app.services.run_store import RunStore

# Matches app.utils.id_utils.generate_run_id: run_YYYY_MM_DD_HH_MM_SS_xxxxxx
_RUN_TIMESTAMP_FORMAT = "run_%Y_%m_%d_%H_%M_%S"


def _parse_run_timestamp(run_dir_name: str) -> datetime | None:
    parts = run_dir_name.split("_")
    if len(parts) < 7:
        return None
    timestamp_part = "_".join(parts[:7])
    try:
        return datetime.strptime(timestamp_part, _RUN_TIMESTAMP_FORMAT)
    except ValueError:
        return None


def run_cleanup(settings: Settings, retention_days: int | None = None, dry_run: bool = False) -> Dict[str, int]:
    """
    Delete data/outputs/run_* folders older than retention_days (defaults to
    settings.output_retention_days). Also removes the corresponding row from
    the SQLite run index. Shared by scripts/cleanup_outputs.py (manual/cron)
    and the automatic scheduled cleanup started in app.main's lifespan.
    """
    days = retention_days if retention_days is not None else settings.output_retention_days

    result = {"deleted": 0, "kept": 0, "unparseable": 0}

    if days <= 0:
        return result

    output_dir = settings.output_dir
    if not output_dir.exists():
        return result

    cutoff = datetime.now() - timedelta(days=days)
    run_store = RunStore(settings)

    for run_dir in sorted(output_dir.iterdir()):
        if not run_dir.is_dir() or not run_dir.name.startswith("run_"):
            continue

        run_timestamp = _parse_run_timestamp(run_dir.name)
        if run_timestamp is None:
            result["unparseable"] += 1
            continue

        if run_timestamp < cutoff:
            if not dry_run:
                shutil.rmtree(run_dir, ignore_errors=True)
                run_store.delete_run(run_dir.name)
            result["deleted"] += 1
        else:
            result["kept"] += 1

    return result

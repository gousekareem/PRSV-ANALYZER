"""
Retention/cleanup job for data/outputs/.

Every analysis run writes a timestamped folder under data/outputs/ (images,
JSON, heatmaps, logs). As of v2.9 this now runs automatically in the
background (see app.main's lifespan, controlled by OUTPUT_RETENTION_DAYS in
.env) - this script remains for manual runs, cron, or Windows Task Scheduler
if you'd rather trigger cleanup outside the running app.

Usage (from prsv_project/):
    python scripts/cleanup_outputs.py                # delete runs older than the configured retention
    python scripts/cleanup_outputs.py --dry-run       # show what would be deleted, delete nothing
    python scripts/cleanup_outputs.py --days 7        # override retention window for this run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import settings
from app.services.cleanup_service import run_cleanup


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--days",
        type=int,
        default=None,
        help="Override OUTPUT_RETENTION_DAYS from settings/.env for this invocation.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be deleted without deleting anything.",
    )
    args = parser.parse_args()

    retention_days = args.days if args.days is not None else settings.output_retention_days

    if retention_days <= 0:
        print("Retention is disabled (output_retention_days <= 0). Nothing to do.")
        return

    result = run_cleanup(settings, retention_days=retention_days, dry_run=args.dry_run)

    label = "[DRY RUN] Would delete" if args.dry_run else "Deleted"
    print(f"{label}: {result['deleted']}   Kept: {result['kept']}   Unparseable folder names skipped: {result['unparseable']}")
    if args.dry_run and result["deleted"]:
        print("Dry run only - nothing was actually deleted. Re-run without --dry-run to apply.")


if __name__ == "__main__":
    main()

"""
Auto-label the demo dataset from filenames as a starting point for training.

The shipped "Original Images" folder already encodes ground truth in its
filenames: Healthy(n).jpg and RingSpot(n).jpg. This script derives labels
from that naming convention and writes them into labels_template.csv, so the
761-image dataset stops being "0 labels filled" and scripts/extract_features_dataset.py
-> ml/train_svm.py can actually run.

This is a *starting point*, not a substitute for expert review: filenames are
only as trustworthy as whoever sorted the files in the first place. Spot-check
a sample of each class before trusting the trained model for anything beyond
a research demo. Any row you manually correct in labels_template.csv will be
respected (the script only fills rows whose label is currently blank, unless
--force is passed).

Usage (from prsv_project/):
    python scripts/auto_label_from_filenames.py
    python scripts/auto_label_from_filenames.py --force   # overwrite existing labels too
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from app.config import settings
from app.services.dataset_service import DatasetService

# Filename prefix -> label. Extend this if new classes/folders are added later.
PREFIX_LABEL_MAP = {
    "healthy": "Healthy",
    "ringspot": "Diseased",
    "ring_spot": "Diseased",
    "ring-spot": "Diseased",
    "prsv": "Diseased",
    "diseased": "Diseased",
}

_PREFIX_PATTERN = re.compile(r"^([a-zA-Z_-]+)")


def infer_label_from_filename(filename: str) -> str | None:
    match = _PREFIX_PATTERN.match(filename)
    if not match:
        return None
    prefix = match.group(1).strip("_- ").lower()
    return PREFIX_LABEL_MAP.get(prefix)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite labels that are already filled in labels_template.csv.",
    )
    args = parser.parse_args()

    labels_path = PROJECT_ROOT / "labels_template.csv"

    if labels_path.exists():
        df = pd.read_csv(labels_path)
        if "filename" not in df.columns or "label" not in df.columns:
            raise ValueError("labels_template.csv must contain filename and label columns.")
    else:
        dataset_service = DatasetService(settings)
        image_paths = dataset_service.list_demo_images()
        df = pd.DataFrame({"filename": [p.name for p in image_paths], "label": ""})

    df["label"] = df["label"].fillna("").astype(str)

    filled = 0
    skipped_existing = 0
    unmatched: list[str] = []

    for idx, row in df.iterrows():
        filename = str(row["filename"])
        current_label = row["label"].strip()

        if current_label and not args.force:
            skipped_existing += 1
            continue

        inferred = infer_label_from_filename(filename)
        if inferred is None:
            unmatched.append(filename)
            continue

        df.at[idx, "label"] = inferred
        filled += 1

    df.to_csv(labels_path, index=False)

    print(f"Labels written to: {labels_path.resolve()}")
    print(f"Rows filled/updated : {filled}")
    print(f"Rows left unchanged : {skipped_existing} (already labeled; use --force to overwrite)")
    if unmatched:
        print(f"Rows with no recognizable prefix (still blank): {len(unmatched)}")
        for name in unmatched[:20]:
            print(f"  - {name}")
        if len(unmatched) > 20:
            print(f"  ... and {len(unmatched) - 20} more")

    counts = df["label"].value_counts(dropna=False)
    print("\nLabel distribution:")
    print(counts.to_string())


if __name__ == "__main__":
    main()

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from app.config import settings
from app.services.dataset_service import DatasetService


def main() -> None:
    dataset_service = DatasetService(settings)
    image_paths = dataset_service.list_demo_images()

    rows = []
    for path in image_paths:
        rows.append(
            {
                "filename": path.name,
                "absolute_path": str(path.resolve()),
                "suffix": path.suffix.lower(),
                "size_bytes": path.stat().st_size,
            }
        )

    output_path = PROJECT_ROOT / "data" / "demo_manifest.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)

    print(f"Manifest created: {output_path.resolve()}")
    print(f"Rows written: {len(df)}")


if __name__ == "__main__":
    main()
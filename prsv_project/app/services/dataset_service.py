from __future__ import annotations

from pathlib import Path
from typing import List

from app.config import Settings


class DatasetService:
    """
    Handles demo dataset discovery and validation.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def demo_dataset_exists(self) -> bool:
        path = self.settings.demo_dataset_path
        return path.exists() and path.is_dir()

    def list_demo_images(self) -> List[Path]:
        if not self.demo_dataset_exists():
            return []

        image_paths: List[Path] = []
        for item in self.settings.demo_dataset_path.iterdir():
            if item.is_file() and item.suffix.lower() in self.settings.allowed_extensions:
                image_paths.append(item)

        return sorted(image_paths)

    def get_demo_sample(self, limit: int = 10) -> List[Path]:
        return self.list_demo_images()[:limit]
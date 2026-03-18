from __future__ import annotations

from functools import lru_cache

from app.config import Settings, settings
from app.services.dataset_service import DatasetService
from app.services.run_manager import RunManager


@lru_cache
def get_settings() -> Settings:
    """
    Cached settings dependency.
    """
    return settings


def get_run_manager() -> RunManager:
    """
    Create a new run manager instance.
    """
    return RunManager(get_settings())


def get_dataset_service() -> DatasetService:
    """
    Create a dataset service instance.
    """
    return DatasetService(get_settings())
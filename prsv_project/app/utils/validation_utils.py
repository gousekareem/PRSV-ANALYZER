from __future__ import annotations

from pathlib import Path
from typing import Iterable

from PIL import Image


def is_allowed_extension(filename: str, allowed_extensions: Iterable[str]) -> bool:
    """
    Check whether a file extension is allowed.
    """
    suffix = Path(filename).suffix.lower()
    return suffix in {ext.lower() for ext in allowed_extensions}


def validate_file_size(file_size_bytes: int, max_size_mb: int) -> bool:
    """
    Validate file size against the configured maximum in MB.
    """
    max_size_bytes = max_size_mb * 1024 * 1024
    return file_size_bytes <= max_size_bytes


def validate_image_readable(path: Path) -> bool:
    """
    Check whether a file is a readable image.
    """
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def validate_non_empty_file(path: Path) -> bool:
    """
    Ensure the file exists and is non-empty.
    """
    return path.exists() and path.is_file() and path.stat().st_size > 0
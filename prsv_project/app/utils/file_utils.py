from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import UploadFile

from app.utils.id_utils import generate_short_id
from app.utils.path_utils import ensure_dir


def sanitize_filename(filename: str) -> str:
    """
    Keep only safe filename characters.
    """
    safe_chars = []
    for char in filename:
        if char.isalnum() or char in ("_", "-", "."):
            safe_chars.append(char)
        else:
            safe_chars.append("_")
    sanitized = "".join(safe_chars).strip("._")
    return sanitized or f"file_{generate_short_id(6)}"


def ensure_unique_path(directory: Path, filename: str) -> Path:
    """
    Prevent collisions by creating a unique filename when needed.
    """
    directory = ensure_dir(directory)
    filename = sanitize_filename(filename)

    candidate = directory / filename
    if not candidate.exists():
        return candidate

    stem = candidate.stem
    suffix = candidate.suffix
    counter = 1

    while True:
        new_candidate = directory / f"{stem}_{counter}{suffix}"
        if not new_candidate.exists():
            return new_candidate
        counter += 1


async def save_upload_file(upload_file: UploadFile, destination_dir: Path) -> Path:
    """
    Save an uploaded file safely to disk.
    """
    destination_path = ensure_unique_path(destination_dir, upload_file.filename or "uploaded_file")
    ensure_dir(destination_dir)

    with destination_path.open("wb") as buffer:
        while True:
            chunk = await upload_file.read(1024 * 1024)
            if not chunk:
                break
            buffer.write(chunk)

    await upload_file.close()
    return destination_path


def copy_file(src: Path, dst_dir: Path, filename: str | None = None) -> Path:
    """
    Copy a file to a destination directory safely.
    """
    ensure_dir(dst_dir)
    target_name = filename if filename else src.name
    destination = ensure_unique_path(dst_dir, target_name)
    shutil.copy2(src, destination)
    return destination
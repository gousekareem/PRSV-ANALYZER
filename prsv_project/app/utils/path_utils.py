from __future__ import annotations

from pathlib import Path
from typing import Iterable


def ensure_dir(path: Path) -> Path:
    """
    Create the directory if it does not exist and return it.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_dirs(paths: Iterable[Path]) -> None:
    """
    Create multiple directories safely.
    """
    for path in paths:
        ensure_dir(path)


def normalize_path(path: str | Path) -> Path:
    """
    Return an absolute normalized Path object.
    """
    return Path(path).expanduser().resolve()


def is_within_directory(base_dir: Path, target_path: Path) -> bool:
    """
    Prevent path traversal by ensuring target_path is inside base_dir.
    """
    try:
        base_dir = base_dir.resolve()
        target_path = target_path.resolve()
        target_path.relative_to(base_dir)
        return True
    except ValueError:
        return False
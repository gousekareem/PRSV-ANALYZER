from __future__ import annotations

from datetime import datetime
from uuid import uuid4


def generate_short_id(length: int = 8) -> str:
    """
    Generate a short random identifier.
    """
    return uuid4().hex[:length]


def generate_run_id() -> str:
    """
    Generate a timestamped run identifier.
    """
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    return f"run_{timestamp}_{generate_short_id(6)}"


def generate_image_id(prefix: str = "img") -> str:
    """
    Generate a unique image identifier.
    """
    return f"{prefix}_{generate_short_id(10)}"
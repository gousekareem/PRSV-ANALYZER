from pathlib import Path

from app.utils.validation_utils import is_allowed_extension


def test_allowed_extension_accepts_jpg() -> None:
    assert is_allowed_extension("leaf.jpg", [".jpg", ".png"])


def test_allowed_extension_rejects_txt() -> None:
    assert not is_allowed_extension("note.txt", [".jpg", ".png"])
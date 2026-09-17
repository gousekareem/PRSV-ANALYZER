from pathlib import Path
from zipfile import ZipFile

from app.utils.zip_utils import safe_extract_zip


def test_safe_extract_zip_filters_supported_files(tmp_path: Path) -> None:
    zip_path = tmp_path / "images.zip"
    with ZipFile(zip_path, "w") as archive:
        archive.writestr("leaf1.jpg", b"fakejpg")
        archive.writestr("notes.txt", b"ignored")

    extract_dir = tmp_path / "extracted"
    files = safe_extract_zip(zip_path, extract_dir, allowed_extensions=[".jpg", ".png"])

    assert len(files) == 1
    assert files[0].suffix.lower() == ".jpg"
from __future__ import annotations

from pathlib import Path
from zipfile import BadZipFile, ZipFile

from app.utils.file_utils import ensure_unique_path, sanitize_filename
from app.utils.path_utils import ensure_dir, is_within_directory


def safe_extract_zip(
    zip_path: Path,
    extract_dir: Path,
    allowed_extensions: list[str] | None = None,
) -> list[Path]:
    """
    Extract a ZIP file safely while preventing path traversal.
    Only allowed image extensions are extracted when allowed_extensions is provided.
    """
    extracted_files: list[Path] = []
    ensure_dir(extract_dir)

    normalized_allowed = {ext.lower() for ext in (allowed_extensions or [])}

    try:
        with ZipFile(zip_path, "r") as archive:
            members = archive.infolist()
            if not members:
                raise ValueError("ZIP archive is empty.")

            for member in members:
                if member.is_dir():
                    continue

                original_name = Path(member.filename).name
                if not original_name:
                    continue

                safe_name = sanitize_filename(original_name)
                suffix = Path(safe_name).suffix.lower()

                if normalized_allowed and suffix not in normalized_allowed:
                    continue

                target_path = ensure_unique_path(extract_dir, safe_name).resolve()

                if not is_within_directory(extract_dir.resolve(), target_path):
                    raise ValueError("Unsafe ZIP content detected.")

                with archive.open(member) as source, target_path.open("wb") as target:
                    target.write(source.read())

                extracted_files.append(target_path)

    except BadZipFile as exc:
        raise ValueError("Invalid or corrupt ZIP archive.") from exc

    if not extracted_files:
        raise ValueError("No supported image files were found in the ZIP archive.")

    return extracted_files
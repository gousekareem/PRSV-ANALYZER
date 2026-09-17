from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.config import Settings
from app.utils.id_utils import generate_run_id
from app.utils.path_utils import ensure_dir


@dataclass
class RunContext:
    run_id: str
    run_dir: Path
    images_dir: Path
    charts_dir: Path
    logs_dir: Path
    log_file: Path


class RunManager:
    """
    Manages timestamped output folders for every run.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def create_run(self) -> RunContext:
        run_id = generate_run_id()
        run_dir = ensure_dir(self.settings.output_dir / run_id)
        images_dir = ensure_dir(run_dir / "images")
        charts_dir = ensure_dir(run_dir / "charts")
        logs_dir = ensure_dir(run_dir / "logs")
        log_file = logs_dir / "run.log"

        return RunContext(
            run_id=run_id,
            run_dir=run_dir,
            images_dir=images_dir,
            charts_dir=charts_dir,
            logs_dir=logs_dir,
            log_file=log_file,
        )

    def create_image_dir(self, run_context: RunContext, image_id: str) -> Path:
        return ensure_dir(run_context.images_dir / image_id)
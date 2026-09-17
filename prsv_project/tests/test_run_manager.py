from app.config import settings
from app.services.run_manager import RunManager


def test_run_context_creation() -> None:
    manager = RunManager(settings)
    context = manager.create_run()

    assert context.run_id.startswith("run_")
    assert context.run_dir.exists()
    assert context.images_dir.exists()
    assert context.logs_dir.exists()
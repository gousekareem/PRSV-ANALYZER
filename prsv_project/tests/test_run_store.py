from pathlib import Path

from app.config import Settings
from app.services.run_store import RunStore


def _make_settings(tmp_path: Path) -> Settings:
    settings = Settings(DEMO_DATASET_PATH=str(tmp_path))
    # data_dir is derived from ROOT_DIR normally; monkeypatch via property override
    settings.__dict__["_test_data_dir"] = tmp_path
    return settings


def test_run_store_upsert_and_search(tmp_path: Path, monkeypatch) -> None:
    settings = Settings(DEMO_DATASET_PATH=str(tmp_path))
    monkeypatch.setattr(type(settings), "data_dir", property(lambda self: tmp_path))

    store = RunStore(settings)
    store.upsert_run(
        run_id="run_2026_01_01_00_00_00_abc123",
        created_at="2026-01-01T00:00:00",
        total_images=5,
        processed_images=5,
        failed_images=0,
        healthy_count=3,
        diseased_count=2,
        average_confidence=0.9,
        average_infection_percentage=12.5,
        filenames=["leaf1.jpg", "leaf2.jpg"],
    )

    result = store.list_runs()
    assert result["total"] == 1
    assert result["items"][0]["run_id"] == "run_2026_01_01_00_00_00_abc123"

    search_hit = store.list_runs(query="leaf1")
    assert search_hit["total"] == 1

    search_miss = store.list_runs(query="nonexistent")
    assert search_miss["total"] == 0


def test_run_store_job_tracking(tmp_path: Path, monkeypatch) -> None:
    settings = Settings(DEMO_DATASET_PATH=str(tmp_path))
    monkeypatch.setattr(type(settings), "data_dir", property(lambda self: tmp_path))

    store = RunStore(settings)
    store.create_job(job_id="job_abc123", created_at="2026-01-01T00:00:00", total_images=3)

    job = store.get_job("job_abc123")
    assert job is not None
    assert job["status"] == "queued"

    store.update_job(job_id="job_abc123", status="done", updated_at="2026-01-01T00:01:00", run_id="run_xyz")
    updated = store.get_job("job_abc123")
    assert updated["status"] == "done"
    assert updated["run_id"] == "run_xyz"

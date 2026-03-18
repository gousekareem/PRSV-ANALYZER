from app.config import settings
from app.services.dataset_service import DatasetService


def test_dataset_service_list_demo_images_returns_list() -> None:
    service = DatasetService(settings)
    result = service.list_demo_images()
    assert isinstance(result, list)
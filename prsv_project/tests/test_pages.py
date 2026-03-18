from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_results_page_loads() -> None:
    response = client.get("/results")
    assert response.status_code == 200


def test_batch_results_page_loads() -> None:
    response = client.get("/batch-results")
    assert response.status_code == 200
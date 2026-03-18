from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_results_page_with_search_query_loads() -> None:
    response = client.get("/results?search=run")
    assert response.status_code == 200


def test_results_page_with_filters_loads() -> None:
    response = client.get("/results?status_filter=diseased&sort_by=newest")
    assert response.status_code == 200
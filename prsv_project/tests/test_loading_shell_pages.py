from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_analyze_page_contains_forms() -> None:
    response = client.get("/analyze")
    assert response.status_code == 200
    text = response.text
    assert 'data-analysis-form="true"' in text
    assert 'data-analysis-mode="single"' in text
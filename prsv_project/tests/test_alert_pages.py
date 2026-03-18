from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_analyze_page_loads_with_message_params() -> None:
    response = client.get("/analyze?message=Test%20message&level=warning")
    assert response.status_code == 200
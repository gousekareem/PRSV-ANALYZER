from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_runs_api_loads() -> None:
    response = client.get("/api/analysis/runs")
    assert response.status_code == 200
    payload = response.json()
    assert "runs" in payload
from io import BytesIO

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_single_form_invalid_extension_redirects() -> None:
    response = client.post(
        "/analyze/single",
        files={"file": ("bad.txt", BytesIO(b"hello"), "text/plain")},
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert "/analyze?message=" in response.headers["location"]


def test_zip_form_invalid_extension_redirects() -> None:
    response = client.post(
        "/analyze/zip",
        files={"file": ("bad.jpg", BytesIO(b"hello"), "image/jpeg")},
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert "/analyze?message=" in response.headers["location"]
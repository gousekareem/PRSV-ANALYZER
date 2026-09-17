from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_analyze_page_contains_forms() -> None:
    """
    v2.9 replaced the old fake-timed-progress overlay (driven by
    data-analysis-form/data-analysis-mode attributes) with real upload/
    processing progress driven by upload.js against XHR + the async job
    queue. This test checks for the new form/dropzone markup instead.
    """
    response = client.get("/analyze")
    assert response.status_code == 200
    text = response.text

    assert 'id="single-upload-form"' in text
    assert 'id="single-dropzone"' in text
    assert 'id="multi-upload-form"' in text
    assert 'id="zip-upload-form"' in text
    assert 'id="demo-analysis-form"' in text
    assert 'capture="environment"' in text

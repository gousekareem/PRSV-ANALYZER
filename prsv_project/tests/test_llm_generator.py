import os

from rag.llm_generator import generate_grounded_explanation, is_configured
from rag.schemas import RetrievedChunk


def test_is_configured_false_without_api_key(monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert is_configured() is False


def test_generate_grounded_explanation_returns_none_without_api_key(monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    result = generate_grounded_explanation(
        prediction="Diseased",
        confidence=0.9,
        severity_label="Moderate",
        severity_score=45.0,
        retrieved_chunks=[
            RetrievedChunk(
                chunk_id="c1",
                title="PRSV Symptoms",
                category="symptoms",
                audience="farmer",
                text="Yellowing and mosaic patterns are common PRSV symptoms.",
                tags=["symptoms"],
                similarity_score=0.9,
            )
        ],
    )

    # No API key configured -> must return None so callers fall back to the
    # deterministic template generator, never raise or hang.
    assert result is None


def test_generate_grounded_explanation_handles_bad_key_gracefully(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "invalid-test-key-not-a-real-key")

    result = generate_grounded_explanation(
        prediction="Healthy",
        confidence=0.95,
        severity_label="Healthy",
        severity_score=0.0,
        retrieved_chunks=[],
    )

    # Either the anthropic package isn't installed, or the call fails against
    # the fake key - both should be swallowed and return None, never raise.
    assert result is None or isinstance(result, str)

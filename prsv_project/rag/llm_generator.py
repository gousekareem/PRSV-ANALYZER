from __future__ import annotations

"""
Real LLM generation for the RAG pipeline (v3.0).

The original rag/generator.py composes explanations by string-templating the
top retrieved chunks together - deterministic and dependency-free, but not
actual generation. This module adds a real generation step: a grounded
Claude API call over the retrieved evidence, with a strict "only use the
provided context" system prompt to keep hallucination risk down.

This is opt-in and fails safe:
- No ANTHROPIC_API_KEY configured, or the `anthropic` package missing, or the
  API call errors/times out -> returns None, and callers (app/services/
  rag_service.py) fall back to the original deterministic template generator.
  The app therefore works identically to v2.9 out of the box, and only
  produces LLM-generated text once an API key is explicitly set.
- The prompt instructs the model to say so explicitly if the retrieved
  context doesn't cover the question, rather than filling gaps from its own
  training knowledge - this is what the RAGAS "faithfulness" metric in
  rag/rag_eval.py checks after the fact.
"""

import os
from typing import List, Optional

from rag.schemas import RetrievedChunk

GENERATION_MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 500

_SYSTEM_PROMPT = (
    "You are an agricultural assistant explaining a Papaya Ring Spot Virus "
    "(PRSV) leaf-image diagnosis to a farmer or an agronomy reviewer. "
    "Base your answer ONLY on the structured findings and retrieved knowledge "
    "chunks provided below. Do not introduce facts, chemical names, or "
    "treatment claims that are not present in the provided context. If the "
    "context does not fully answer something, say so plainly instead of "
    "guessing. Keep the answer concise, practical, and non-alarmist."
)


def _format_context(
    prediction: str,
    confidence: float,
    severity_label: str,
    severity_score: float,
    retrieved_chunks: List[RetrievedChunk],
) -> str:
    chunk_lines = "\n".join(
        f"- [{c.title}] {c.text}" for c in retrieved_chunks
    ) or "(no knowledge chunks retrieved)"

    return (
        f"Structured findings:\n"
        f"- Prediction: {prediction}\n"
        f"- Confidence: {confidence:.4f}\n"
        f"- Severity: {severity_label} ({severity_score:.1f}/100)\n\n"
        f"Retrieved PRSV knowledge:\n{chunk_lines}"
    )


def is_configured() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def generate_grounded_explanation(
    prediction: str,
    confidence: float,
    severity_label: str,
    severity_score: float,
    retrieved_chunks: List[RetrievedChunk],
    audience: str = "farmer",
) -> Optional[str]:
    """
    Attempt a real, grounded LLM generation. Returns None on any failure so
    the caller can fall back to the deterministic template generator.
    """
    if not is_configured():
        return None

    try:
        import anthropic

        client = anthropic.Anthropic()
        context = _format_context(prediction, confidence, severity_label, severity_score, retrieved_chunks)
        audience_instruction = (
            "Write for a farmer with no technical background: short sentences, plain language, no jargon."
            if audience == "farmer"
            else "Write for an agronomy reviewer: precise, technical language is fine."
        )

        response = client.messages.create(
            model=GENERATION_MODEL,
            max_tokens=MAX_TOKENS,
            system=_SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": f"{audience_instruction}\n\n{context}\n\nExplain this diagnosis and what to do next.",
                }
            ],
        )
        text_blocks = [block.text for block in response.content if getattr(block, "type", None) == "text"]
        combined = "\n".join(text_blocks).strip()
        return combined or None
    except Exception:  # noqa: BLE001 - generation is best-effort, template fallback always available
        return None

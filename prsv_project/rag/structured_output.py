from __future__ import annotations

"""
Structured / constrained generation (v3.1): forces the LLM's output into a
fixed schema (diagnosis summary + prioritized action list + confidence
caveat) instead of free text, for more reliable downstream parsing by a
frontend or another service.

Approach: rather than a generic "constrained decoding" library (outlines/
guidance need direct control of token sampling, which the Anthropic Messages
API doesn't expose), this uses Claude's structured-output-via-tool-use
pattern - defining a single-purpose "tool" whose input schema IS the desired
output shape, and forcing the model to call it. This is the standard,
API-native way to get schema-validated JSON out of Claude, and degrades the
same way the rest of rag/ does: returns None on any failure (no key, package
missing, API error, schema validation failure) so callers fall back to the
existing free-text generator.
"""

from typing import List, Optional

from pydantic import BaseModel, Field

from rag.llm_generator import is_configured
from rag.schemas import RetrievedChunk

_OUTPUT_TOOL_NAME = "emit_diagnosis_summary"

_OUTPUT_TOOL_SCHEMA = {
    "name": _OUTPUT_TOOL_NAME,
    "description": "Emit a structured PRSV diagnosis summary for the farmer/reviewer.",
    "input_schema": {
        "type": "object",
        "properties": {
            "summary": {"type": "string", "description": "One or two plain-language sentences summarizing the diagnosis."},
            "actions": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Prioritized, concrete next steps, grounded only in the provided context.",
            },
            "confidence_caveat": {
                "type": "string",
                "description": "A short caveat about how confident this diagnosis is and what could change it.",
            },
        },
        "required": ["summary", "actions", "confidence_caveat"],
    },
}


class StructuredDiagnosis(BaseModel):
    summary: str
    actions: List[str] = Field(default_factory=list)
    confidence_caveat: str


def generate_structured_diagnosis(
    prediction: str,
    confidence: float,
    severity_label: str,
    severity_score: float,
    retrieved_chunks: List[RetrievedChunk],
) -> Optional[StructuredDiagnosis]:
    if not is_configured():
        return None

    try:
        import anthropic

        client = anthropic.Anthropic()
        context_lines = "\n".join(f"- [{c.title}] {c.text}" for c in retrieved_chunks) or "(no context retrieved)"

        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=400,
            tools=[_OUTPUT_TOOL_SCHEMA],
            tool_choice={"type": "tool", "name": _OUTPUT_TOOL_NAME},
            system=(
                "You produce structured PRSV diagnosis summaries. Base your output ONLY "
                "on the structured findings and context given; never introduce facts not present there."
            ),
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Prediction: {prediction} (confidence {confidence:.4f})\n"
                        f"Severity: {severity_label} ({severity_score:.1f}/100)\n"
                        f"Context:\n{context_lines}"
                    ),
                }
            ],
        )

        for block in response.content:
            if getattr(block, "type", None) == "tool_use" and block.name == _OUTPUT_TOOL_NAME:
                return StructuredDiagnosis(**block.input)
        return None
    except Exception:  # noqa: BLE001 - best-effort (incl. ValidationError), template fallback always available
        return None

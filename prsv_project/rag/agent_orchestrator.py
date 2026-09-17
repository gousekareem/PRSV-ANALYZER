from __future__ import annotations

"""
Agentic RAG orchestration (v3.1): a router that decides, per incoming
chatbot message, whether to call the image-analysis pipeline, the KB
retriever, or answer directly - instead of chatbot_service.py's current
fixed flow (always: check greeting regex, else always retrieve+answer).

Honesty note on what "agent" means here: a full LangChain/LlamaIndex agent
loop lets an LLM freely decide which of N tools to call, in what order, for
how many steps, based on its own reasoning. Wiring that up meaningfully
needs an LLM call for the routing decision itself, which costs latency and
money per message just to route. What's implemented below is a real,
testable, and honest middle ground: a deterministic router when no LLM is
configured (which is what most of this project's traffic looks like, since
ANTHROPIC_API_KEY is opt-in - see rag/llm_generator.py), and a genuine
LLM-driven tool-choice call (Claude tool use) as a strictly-better upgrade
path when a key IS configured. Both paths converge on the same
ORCHESTRATION_TOOLS action space, so swapping between them doesn't change
what the caller (chatbot_service.py) sees.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from rag.llm_generator import is_configured as llm_is_configured

_HAS_PHOTO_HINTS = {"photo", "picture", "image", "attach", "upload", "check this"}
_HAS_ANALYSIS_KEYWORDS = {"diagnose", "diagnosis", "what disease", "is this prsv", "check my plant"}


class OrchestratorAction(str, Enum):
    ANALYZE_PHOTO = "analyze_photo"
    RETRIEVE_KNOWLEDGE = "retrieve_knowledge"
    DIRECT_ANSWER = "direct_answer"


@dataclass
class OrchestratorDecision:
    action: OrchestratorAction
    reason: str
    method: str  # "rule_based" or "llm_tool_use"


_TOOLS_SCHEMA = [
    {
        "name": "analyze_photo",
        "description": "Run the leaf-image analysis pipeline. Use when the farmer has attached/mentioned a photo or explicitly asks for a diagnosis of a specific plant.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "retrieve_knowledge",
        "description": "Answer using the PRSV knowledge base. Use for general questions about symptoms, causes, treatment, prevention.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "direct_answer",
        "description": "Answer directly without retrieval - use only for greetings, thanks, or small talk with no PRSV content.",
        "input_schema": {"type": "object", "properties": {}},
    },
]


def _rule_based_route(message: str, has_attached_photo: bool) -> OrchestratorDecision:
    if has_attached_photo:
        return OrchestratorDecision(OrchestratorAction.ANALYZE_PHOTO, "A photo was attached to this message.", "rule_based")

    lowered = message.lower()

    if any(kw in lowered for kw in _HAS_ANALYSIS_KEYWORDS) or any(kw in lowered for kw in _HAS_PHOTO_HINTS):
        return OrchestratorDecision(
            OrchestratorAction.ANALYZE_PHOTO,
            "Message mentions a photo/diagnosis request but none is attached yet - prompts the user to attach one.",
            "rule_based",
        )

    if len(lowered.split()) <= 2 and not any(c.isalpha() and len(w) > 4 for w in lowered.split() for c in w):
        return OrchestratorDecision(OrchestratorAction.DIRECT_ANSWER, "Very short message, likely greeting/small talk.", "rule_based")

    return OrchestratorDecision(OrchestratorAction.RETRIEVE_KNOWLEDGE, "Default: treat as a PRSV knowledge question.", "rule_based")


def _llm_route(message: str, has_attached_photo: bool) -> Optional[OrchestratorDecision]:
    try:
        import anthropic

        client = anthropic.Anthropic()
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=50,
            tools=_TOOLS_SCHEMA,
            tool_choice={"type": "any"},
            system=(
                "Route this farmer chatbot message to exactly one tool. "
                f"A photo is {'already attached' if has_attached_photo else 'NOT attached'} to this message."
            ),
            messages=[{"role": "user", "content": message}],
        )
        for block in response.content:
            if getattr(block, "type", None) == "tool_use":
                return OrchestratorDecision(OrchestratorAction(block.name), "LLM tool-use routing decision.", "llm_tool_use")
        return None
    except Exception:  # noqa: BLE001 - routing is best-effort, rule-based fallback always available
        return None


def route(message: str, has_attached_photo: bool = False) -> OrchestratorDecision:
    if llm_is_configured():
        llm_decision = _llm_route(message, has_attached_photo)
        if llm_decision is not None:
            return llm_decision
    return _rule_based_route(message, has_attached_photo)

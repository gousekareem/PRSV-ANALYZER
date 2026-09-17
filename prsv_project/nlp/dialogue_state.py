from __future__ import annotations

"""
Multi-turn dialogue state tracking (v3.1): remembers "the plant we're
discussing" and recent topic across turns within one chat session, plus a
simple coreference heuristic for "it", "that plant", "the same one".

chatbot_service.py currently treats every message independently (each
handle_message() call is stateless). This module provides the session-scoped
memory structure; wiring it into chatbot_service.py's request handling is
the next integration step (it needs a session_id concept in the API layer,
which is a routing change, not just a service-layer addition - noted here
rather than silently left out).
"""

from dataclasses import dataclass, field
from typing import List, Optional

_COREFERENCE_PRONOUNS = {"it", "that", "this", "the same one", "same plant", "that plant", "this plant"}


@dataclass
class Turn:
    text: str
    intent: Optional[str] = None
    mentioned_topic: Optional[str] = None


@dataclass
class DialogueState:
    session_id: str
    turns: List[Turn] = field(default_factory=list)
    last_topic: Optional[str] = None
    last_image_id: Optional[str] = None

    def add_turn(self, text: str, intent: Optional[str] = None, mentioned_topic: Optional[str] = None) -> None:
        self.turns.append(Turn(text=text, intent=intent, mentioned_topic=mentioned_topic))
        if mentioned_topic:
            self.last_topic = mentioned_topic

    def resolve_coreference(self, text: str) -> str:
        """
        Replace a bare pronoun reference to "the plant/topic we were just
        discussing" with the actual last-known topic, so downstream
        retrieval gets a query with real content words instead of "is it
        contagious?" with no antecedent.
        """
        lowered = text.lower()
        if not self.last_topic:
            return text

        for pronoun in sorted(_COREFERENCE_PRONOUNS, key=len, reverse=True):
            if pronoun in lowered:
                return lowered.replace(pronoun, self.last_topic, 1)
        return text

    def recent_context_summary(self, max_turns: int = 3) -> str:
        recent = self.turns[-max_turns:]
        return " | ".join(t.text for t in recent)


class DialogueStateStore:
    """
    In-process session store (dict keyed by session_id). For a
    single-machine deployment this is sufficient; a multi-worker deployment
    would need this backed by the same Redis/in-memory CacheService already
    used elsewhere (app/services/cache_service.py) - noted as the natural
    extension point rather than duplicated here.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, DialogueState] = {}

    def get_or_create(self, session_id: str) -> DialogueState:
        if session_id not in self._sessions:
            self._sessions[session_id] = DialogueState(session_id=session_id)
        return self._sessions[session_id]

    def clear(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

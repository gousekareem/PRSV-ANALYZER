from __future__ import annotations

"""
Corrective RAG / CRAG (v3.1): grades whether retrieved chunks are actually
relevant to the query before generation, rather than the current
rag_service.py behavior of only falling back to the template generator on
API/key failure regardless of retrieval quality. If retrieval quality is
judged poor, this signals the caller to either re-query with an expanded
query (see rag/query_expansion.py) or fall back to a safe "I don't have
specific information" response instead of generating over irrelevant
context - directly reducing the risk of a fluent-but-wrong answer.

Grading approach: reuses the same embedding-similarity-or-token-overlap
scoring already implemented in rag/rag_eval.py (no new dependency), applied
per-chunk against the query, since a full LLM-based relevance grader would
cost an extra API call per turn just to decide whether to make a second one.
"""

from dataclasses import dataclass
from typing import List

from rag.rag_eval import _similarity  # reuse the same embedding/overlap scorer
from rag.schemas import RetrievedChunk


@dataclass
class CragDecision:
    action: str  # "generate", "requery", "fallback"
    relevant_chunks: List[RetrievedChunk]
    relevance_scores: List[float]
    reason: str


def grade_and_decide(
    query: str,
    retrieved_chunks: List[RetrievedChunk],
    relevance_threshold: float = 0.15,
    min_relevant_fraction: float = 0.34,
) -> CragDecision:
    if not retrieved_chunks:
        return CragDecision(
            action="fallback",
            relevant_chunks=[],
            relevance_scores=[],
            reason="No chunks were retrieved at all.",
        )

    scores = [_similarity(c.text, query) for c in retrieved_chunks]
    relevant = [c for c, s in zip(retrieved_chunks, scores) if s >= relevance_threshold]
    relevant_fraction = len(relevant) / len(retrieved_chunks)

    if relevant_fraction >= min_relevant_fraction and relevant:
        return CragDecision(
            action="generate",
            relevant_chunks=relevant,
            relevance_scores=scores,
            reason=f"{len(relevant)}/{len(retrieved_chunks)} retrieved chunks judged relevant (>= threshold).",
        )

    if relevant:
        # Some signal, but weak - still worth generating from the relevant
        # subset rather than discarding everything.
        return CragDecision(
            action="generate",
            relevant_chunks=relevant,
            relevance_scores=scores,
            reason="Only a minority of chunks were relevant; generating from that subset only.",
        )

    return CragDecision(
        action="fallback",
        relevant_chunks=[],
        relevance_scores=scores,
        reason="No retrieved chunk cleared the relevance threshold for this query.",
    )

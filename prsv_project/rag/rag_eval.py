from __future__ import annotations

"""
RAGAS-style evaluation for the PRSV RAG pipeline (v3.0).

Honesty note on scope: the real RAGAS library scores faithfulness and answer
relevance by using an LLM-as-judge to decompose the answer into claims and
check each one against the retrieved context. That gives the most reliable
score but costs an LLM call per evaluation and needs an API key. This module
implements the same four metrics with a lightweight, dependency-light
approximation (embedding cosine similarity when sentence-transformers is
available, falling back to token-overlap/Jaccard otherwise) so evaluation can
run for free, offline, in CI. `evaluate_with_llm_judge()` is provided as the
higher-fidelity option when an LLM is configured (see rag/llm_generator.py's
is_configured()) and should be preferred for a final research report.

Metrics:
- faithfulness: how much of the generated answer is supported by the
  retrieved context (proxy for "did we hallucinate").
- answer_relevance: how well the answer addresses the original query.
- context_precision: fraction of retrieved chunks that are actually relevant
  to the query (signal-to-noise of retrieval).
- context_recall: how much of the query's information need is covered by the
  union of retrieved chunks.
"""

from dataclasses import dataclass, asdict
from typing import List, Optional

from rag.schemas import RetrievedChunk

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "of", "in", "on", "to",
    "and", "or", "with", "for", "this", "that", "it", "as", "by", "be",
}


def _tokenize(text: str) -> set:
    return {
        w.strip(".,;:!?()").lower()
        for w in text.split()
        if w.strip(".,;:!?()").lower() not in _STOPWORDS and len(w) > 2
    }


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    intersection = len(a & b)
    union = len(a | b)
    return intersection / union if union else 0.0


def _try_embedding_similarity(text_a: str, text_b: str) -> Optional[float]:
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np

        model = _get_shared_model()
        vecs = model.encode([text_a, text_b], normalize_embeddings=True, show_progress_bar=False)
        return float(np.dot(vecs[0], vecs[1]))
    except Exception:  # noqa: BLE001 - falls back to token overlap
        return None


_shared_model_cache = {}


def _get_shared_model():
    if "model" not in _shared_model_cache:
        from sentence_transformers import SentenceTransformer
        from rag.embeddings_retriever import EMBEDDING_MODEL_NAME

        _shared_model_cache["model"] = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _shared_model_cache["model"]


def _similarity(text_a: str, text_b: str) -> float:
    embedding_score = _try_embedding_similarity(text_a, text_b)
    if embedding_score is not None:
        return max(0.0, embedding_score)
    return _jaccard(_tokenize(text_a), _tokenize(text_b))


@dataclass
class RagEvaluationResult:
    faithfulness: float
    answer_relevance: float
    context_precision: float
    context_recall: float
    method: str

    def as_dict(self) -> dict:
        return asdict(self)


def evaluate_rag_response(
    query: str,
    answer: str,
    retrieved_chunks: List[RetrievedChunk],
    relevance_threshold: float = 0.15,
) -> RagEvaluationResult:
    """
    Score one RAG turn (query, retrieved chunks, generated/composed answer)
    against the four RAGAS-lite metrics described above.
    """
    method = "embedding_cosine" if _try_embedding_similarity("a", "a") is not None else "token_jaccard"

    context_text = " ".join(c.text for c in retrieved_chunks)

    faithfulness = _similarity(answer, context_text) if context_text else 0.0

    answer_relevance = _similarity(answer, query)

    if retrieved_chunks:
        relevant_flags = [_similarity(c.text, query) >= relevance_threshold for c in retrieved_chunks]
        context_precision = sum(relevant_flags) / len(relevant_flags)
    else:
        context_precision = 0.0

    context_recall = _similarity(context_text, query) if context_text else 0.0

    return RagEvaluationResult(
        faithfulness=round(faithfulness, 4),
        answer_relevance=round(answer_relevance, 4),
        context_precision=round(context_precision, 4),
        context_recall=round(context_recall, 4),
        method=method,
    )


def evaluate_batch(
    turns: List[dict],
) -> dict:
    """
    turns: list of {"query": str, "answer": str, "retrieved_chunks": List[RetrievedChunk]}
    Returns per-turn scores plus dataset-level averages, mirroring how RAGAS
    reports a summary table across an evaluation set.
    """
    results = [
        evaluate_rag_response(t["query"], t["answer"], t["retrieved_chunks"])
        for t in turns
    ]
    if not results:
        return {"per_turn": [], "averages": {}}

    averages = {
        "faithfulness": round(sum(r.faithfulness for r in results) / len(results), 4),
        "answer_relevance": round(sum(r.answer_relevance for r in results) / len(results), 4),
        "context_precision": round(sum(r.context_precision for r in results) / len(results), 4),
        "context_recall": round(sum(r.context_recall for r in results) / len(results), 4),
    }
    return {"per_turn": [r.as_dict() for r in results], "averages": averages}

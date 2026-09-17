from __future__ import annotations

"""
Cross-encoder reranking (v3.1) - a precision-boosting second pass over
rag/hybrid_retriever.py's fused BM25+dense results, using a cross-encoder
(a model that scores a (query, passage) pair jointly, rather than encoding
them separately like the bi-encoder in rag/embeddings_retriever.py). Cross-
encoders are slower per-pair but meaningfully more accurate at "is this
specific passage the best match for this specific query," which is exactly
what a small top-K candidate list benefits from.

Best-effort, same degrade pattern as the rest of rag/: if
`sentence-transformers` (which ships CrossEncoder) isn't installed, or the
model fails to load, `rerank()` returns the input list unchanged and ordered
by its original fused score.
"""

from typing import List, Optional

from rag.schemas import RetrievedChunk

RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_reranker_cache: dict = {}


def _get_reranker():
    if "model" not in _reranker_cache:
        try:
            from sentence_transformers import CrossEncoder

            _reranker_cache["model"] = CrossEncoder(RERANKER_MODEL_NAME)
        except Exception:  # noqa: BLE001 - reranking is a best-effort quality boost
            _reranker_cache["model"] = None
    return _reranker_cache["model"]


def is_available() -> bool:
    return _get_reranker() is not None


def rerank(query: str, candidates: List[RetrievedChunk], top_k: Optional[int] = None) -> List[RetrievedChunk]:
    """
    Re-score `candidates` against `query` with a cross-encoder and return
    them sorted by that score (highest first), truncated to top_k. If the
    cross-encoder isn't available, returns `candidates` unchanged - the
    caller's original fused-score ordering is a perfectly reasonable
    fallback ranking on its own.
    """
    model = _get_reranker()
    if model is None or not candidates:
        return candidates[:top_k] if top_k else candidates

    try:
        pairs = [(query, c.text) for c in candidates]
        scores = model.predict(pairs)

        reranked = [
            RetrievedChunk(
                chunk_id=c.chunk_id,
                title=c.title,
                category=c.category,
                audience=c.audience,
                text=c.text,
                tags=c.tags,
                similarity_score=round(float(score), 6),
                retrieval_method=f"{c.retrieval_method}+reranked",
                sparse_rank=c.sparse_rank,
                dense_rank=c.dense_rank,
            )
            for c, score in zip(candidates, scores)
        ]
        reranked.sort(key=lambda c: c.similarity_score, reverse=True)
        return reranked[:top_k] if top_k else reranked
    except Exception:  # noqa: BLE001
        return candidates[:top_k] if top_k else candidates

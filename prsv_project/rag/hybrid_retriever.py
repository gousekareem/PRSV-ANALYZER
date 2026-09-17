from __future__ import annotations

"""
Hybrid sparse + dense retrieval for the PRSV knowledge base (v3.0).

Combines BM25 (exact keyword recall - important for chemical/product names a
farmer might type verbatim, e.g. "imidacloprid") with dense embedding
retrieval (semantic recall - handles paraphrase and, after translation to
English, cross-lingual queries) via Reciprocal Rank Fusion (RRF), which is
the standard, parameter-light way to combine two differently-scaled ranking
signals without having to hand-tune a score weighting.

Degrades gracefully in three stages, in keeping with this project's existing
"never fatal, always answer something" style (see ml/shap_explainer.py):
  1. rank_bm25 + sentence-transformers + faiss all available -> full hybrid RRF
  2. only one of {bm25, dense} available -> use that one alone
  3. neither available -> fall back to the original TF-IDF retriever
     (rag/retriever.py), so this module is a strict upgrade, never a
     regression, on any machine including ones without the new optional deps.
"""

from typing import Dict, List

from rag.embeddings_retriever import DenseEmbeddingRetriever
from rag.retriever import LocalTfidfRetriever
from rag.schemas import KnowledgeChunk, RetrievedChunk

RRF_K = 60  # standard RRF damping constant from the original RRF paper (Cormack et al.)


class _Bm25Sparse:
    def __init__(self, chunks: List[KnowledgeChunk]) -> None:
        self.chunks = chunks
        self._bm25 = None
        try:
            from rank_bm25 import BM25Okapi

            tokenized = [self._tokenize(self._document(c)) for c in chunks]
            self._bm25 = BM25Okapi(tokenized)
        except Exception:  # noqa: BLE001 - optional dependency
            self._bm25 = None

    @staticmethod
    def _document(chunk: KnowledgeChunk) -> str:
        return f"{chunk.title} {chunk.category} {' '.join(chunk.tags)} {chunk.text}"

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return text.lower().split()

    def is_available(self) -> bool:
        return self._bm25 is not None

    def retrieve(self, query: str, top_k: int) -> List[RetrievedChunk]:
        if not self.is_available():
            return []
        scores = self._bm25.get_scores(self._tokenize(query))
        ranked_indices = scores.argsort()[::-1][:top_k]
        results: List[RetrievedChunk] = []
        for rank, idx in enumerate(ranked_indices):
            chunk = self.chunks[int(idx)]
            results.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    title=chunk.title,
                    category=chunk.category,
                    audience=chunk.audience,
                    text=chunk.text,
                    tags=chunk.tags,
                    similarity_score=round(float(scores[idx]), 6),
                    retrieval_method="bm25",
                    sparse_rank=rank + 1,
                )
            )
        return results


class HybridRetriever:
    """
    Hybrid BM25 + dense-embedding retriever with Reciprocal Rank Fusion,
    reranked by a fused score, plus an automatic fallback chain.
    """

    def __init__(self, chunks: List[KnowledgeChunk]) -> None:
        self.chunks = chunks
        self._sparse = _Bm25Sparse(chunks)
        self._dense = DenseEmbeddingRetriever(chunks)
        self._tfidf_fallback = LocalTfidfRetriever(chunks)

    @property
    def active_mode(self) -> str:
        if self._sparse.is_available() and self._dense.is_available():
            return "hybrid_rrf"
        if self._dense.is_available():
            return "dense_only"
        if self._sparse.is_available():
            return "bm25_only"
        return "tfidf_fallback"

    def retrieve(self, query: str, top_k: int = 3, candidate_pool: int = 15) -> List[RetrievedChunk]:
        mode = self.active_mode

        if mode == "tfidf_fallback":
            return self._tfidf_fallback.retrieve(query, top_k=top_k)

        sparse_hits = self._sparse.retrieve(query, top_k=candidate_pool) if self._sparse.is_available() else []
        dense_hits = self._dense.retrieve(query, top_k=candidate_pool) if self._dense.is_available() else []

        if mode == "bm25_only":
            return sparse_hits[:top_k]
        if mode == "dense_only":
            return dense_hits[:top_k]

        return self._fuse(sparse_hits, dense_hits, top_k=top_k)

    def _fuse(
        self,
        sparse_hits: List[RetrievedChunk],
        dense_hits: List[RetrievedChunk],
        top_k: int,
    ) -> List[RetrievedChunk]:
        rrf_scores: Dict[str, float] = {}
        by_id: Dict[str, RetrievedChunk] = {}

        for hit in sparse_hits:
            rrf_scores[hit.chunk_id] = rrf_scores.get(hit.chunk_id, 0.0) + 1.0 / (RRF_K + (hit.sparse_rank or 999))
            by_id[hit.chunk_id] = hit

        for hit in dense_hits:
            rrf_scores[hit.chunk_id] = rrf_scores.get(hit.chunk_id, 0.0) + 1.0 / (RRF_K + (hit.dense_rank or 999))
            if hit.chunk_id in by_id:
                by_id[hit.chunk_id].dense_rank = hit.dense_rank
            else:
                by_id[hit.chunk_id] = hit

        ranked_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)[:top_k]

        fused: List[RetrievedChunk] = []
        for cid in ranked_ids:
            hit = by_id[cid]
            fused.append(
                RetrievedChunk(
                    chunk_id=hit.chunk_id,
                    title=hit.title,
                    category=hit.category,
                    audience=hit.audience,
                    text=hit.text,
                    tags=hit.tags,
                    similarity_score=round(rrf_scores[cid], 6),
                    retrieval_method="hybrid_rrf",
                    sparse_rank=hit.sparse_rank,
                    dense_rank=hit.dense_rank,
                )
            )
        return fused

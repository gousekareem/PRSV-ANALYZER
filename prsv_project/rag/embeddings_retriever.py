from __future__ import annotations

"""
Dense embedding retriever for the PRSV knowledge base (v3.0).

Upgrades the original TF-IDF-only retrieval path (rag/retriever.py) with
multilingual sentence embeddings, so retrieval quality no longer depends on
exact keyword overlap - a farmer's paraphrase, or a query in Hindi/Telugu
after translation, still lands close to the right chunk in embedding space.

Design notes / honesty about trade-offs:
- Uses `sentence-transformers` with a small multilingual model
  (paraphrase-multilingual-MiniLM-L12-v2, ~118M params, CPU-friendly) rather
  than a larger model like bge-m3, so it stays usable on the same modest
  hardware the rest of this project targets. Swapping the model name is a
  one-line change (EMBEDDING_MODEL_NAME below) if more accuracy is needed.
- Uses FAISS (IndexFlatIP over L2-normalized vectors = cosine similarity) when
  installed, and falls back to a plain numpy matrix multiply otherwise. At the
  scale of this knowledge base (tens to low hundreds of chunks) the numpy path
  is not meaningfully slower - FAISS mainly pays off once the KB grows past a
  few thousand chunks, which is exactly the trigger condition called out in
  the upgrade wishlist.
- Everything here is optional-dependency-guarded: if sentence-transformers or
  faiss aren't installed, `is_available()` returns False and callers (see
  rag/hybrid_retriever.py) fall back to the existing TF-IDF retriever alone,
  the same "best-effort, never fatal" pattern already used for SHAP in
  ml/shap_explainer.py.
"""

from pathlib import Path
from typing import List, Optional

import numpy as np

from rag.schemas import KnowledgeChunk, RetrievedChunk

EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


def _chunk_to_document(chunk: KnowledgeChunk) -> str:
    return f"{chunk.title}. {chunk.category}. {' '.join(chunk.tags)}. {chunk.text}"


class DenseEmbeddingRetriever:
    """
    Embedding-based retriever over the PRSV knowledge base.

    Lazily loads the sentence-transformer model and builds the index on first
    use so importing this module never fails just because the optional
    dependency is missing - callers should check `is_available()` first.
    """

    def __init__(self, chunks: List[KnowledgeChunk], cache_dir: Optional[Path] = None) -> None:
        self.chunks = chunks
        self.cache_dir = cache_dir
        self._model = None
        self._index = None
        self._embeddings: Optional[np.ndarray] = None
        self._backend = "uninitialized"
        self._load_error: Optional[str] = None
        self._build()

    def _build(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(EMBEDDING_MODEL_NAME)
            documents = [_chunk_to_document(c) for c in self.chunks]
            embeddings = self._model.encode(
                documents,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            ).astype(np.float32)
            self._embeddings = embeddings

            try:
                import faiss

                dim = embeddings.shape[1]
                index = faiss.IndexFlatIP(dim)
                index.add(embeddings)
                self._index = index
                self._backend = "faiss"
            except Exception:  # noqa: BLE001 - FAISS is an optional accelerator
                self._backend = "numpy"
        except Exception as exc:  # noqa: BLE001 - embeddings are best-effort
            self._load_error = str(exc)
            self._backend = "unavailable"

    def is_available(self) -> bool:
        return self._backend in {"faiss", "numpy"} and self._embeddings is not None

    @property
    def backend(self) -> str:
        return self._backend

    def encode_query(self, query: str) -> Optional[np.ndarray]:
        if not self.is_available() or self._model is None:
            return None
        vector = self._model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
        ).astype(np.float32)
        return vector

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievedChunk]:
        if not self.is_available():
            return []

        query_vector = self.encode_query(query)
        if query_vector is None:
            return []

        top_k = min(top_k, len(self.chunks))

        if self._backend == "faiss":
            scores, indices = self._index.search(query_vector, top_k)
            scores, indices = scores[0], indices[0]
        else:
            similarities = (self._embeddings @ query_vector[0]).astype(np.float32)
            indices = np.argsort(similarities)[::-1][:top_k]
            scores = similarities[indices]

        results: List[RetrievedChunk] = []
        for rank, (idx, score) in enumerate(zip(indices, scores)):
            if idx < 0:
                continue
            chunk = self.chunks[int(idx)]
            results.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    title=chunk.title,
                    category=chunk.category,
                    audience=chunk.audience,
                    text=chunk.text,
                    tags=chunk.tags,
                    similarity_score=round(float(score), 6),
                    retrieval_method="dense",
                    dense_rank=rank + 1,
                )
            )
        return results

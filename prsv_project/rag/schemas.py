from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class KnowledgeChunk:
    chunk_id: str
    title: str
    category: str
    audience: str
    text: str
    tags: List[str]


@dataclass
class RetrievedChunk:
    chunk_id: str
    title: str
    category: str
    audience: str
    text: str
    tags: List[str]
    similarity_score: float
    # v3.0: which retrieval path produced this hit. Defaults keep every existing
    # caller (TF-IDF-only retriever, existing tests) working unchanged.
    retrieval_method: str = "tfidf"
    sparse_rank: int | None = None
    dense_rank: int | None = None
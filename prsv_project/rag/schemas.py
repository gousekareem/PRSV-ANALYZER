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
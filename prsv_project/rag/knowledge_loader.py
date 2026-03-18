from __future__ import annotations

from pathlib import Path
from typing import List

from app.utils.json_utils import load_json
from rag.schemas import KnowledgeChunk


def load_knowledge_base(kb_path: Path) -> List[KnowledgeChunk]:
    """
    Load PRSV knowledge chunks from local JSON knowledge base.
    """
    if not kb_path.exists():
        raise FileNotFoundError(f"Knowledge base file not found: {kb_path}")

    data = load_json(kb_path)

    chunks: List[KnowledgeChunk] = []
    for item in data.get("chunks", []):
        chunks.append(
            KnowledgeChunk(
                chunk_id=str(item["chunk_id"]),
                title=str(item["title"]),
                category=str(item["category"]),
                audience=str(item["audience"]),
                text=str(item["text"]),
                tags=[str(tag) for tag in item.get("tags", [])],
            )
        )

    return chunks
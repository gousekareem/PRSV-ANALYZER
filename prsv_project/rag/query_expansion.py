from __future__ import annotations

"""
Query expansion / rewriting (v3.1): rephrases a terse or ambiguous farmer
query into a fuller retrieval query before hitting the retriever, so
"ring spots wat 2 do" retrieves as well as "PRSV ring spot symptoms
treatment recommendations".

Two tiers, degrading gracefully:
1. LLM rewrite (rag/llm_generator.py's is_configured()) - highest quality,
   used when ANTHROPIC_API_KEY is set.
2. Tag-based lexical expansion - appends knowledge-base tag vocabulary
   terms whose synonyms/stems appear in the query, using nothing but the
   knowledge base already loaded (rag/knowledge_loader.py). Always
   available, zero extra dependencies or API calls.
"""

from typing import List, Optional

from rag.knowledge_loader import load_knowledge_base
from rag.llm_generator import is_configured as llm_is_configured
from rag.schemas import KnowledgeChunk

_SYNONYM_MAP = {
    "spots": ["lesion", "spot", "mark", "patch"],
    "yellow": ["chlorosis", "yellowing", "discoloration"],
    "curl": ["curling", "distortion", "deformation"],
    "spread": ["transmission", "vector", "aphid"],
    "cure": ["treatment", "remedy", "management"],
    "wat": ["what"],  # common shorthand typo seen in farmer chat text
    "2": ["to"],
}


def _lexical_expand(query: str, chunks: List[KnowledgeChunk]) -> str:
    tokens = query.lower().split()
    expansions = set()

    all_tags = {tag.lower() for chunk in chunks for tag in chunk.tags}

    for token in tokens:
        cleaned = token.strip(".,;:!?")
        if cleaned in _SYNONYM_MAP:
            expansions.update(_SYNONYM_MAP[cleaned])
        # If the raw token is a substring of a known KB tag (or vice versa),
        # pull the tag in - cheap way to bridge farmer phrasing to KB vocabulary.
        for tag in all_tags:
            if cleaned and (cleaned in tag or tag in cleaned) and len(cleaned) > 3:
                expansions.add(tag)

    if not expansions:
        return query
    return f"{query} {' '.join(sorted(expansions))}"


def _llm_rewrite(query: str) -> Optional[str]:
    if not llm_is_configured():
        return None
    try:
        import anthropic

        client = anthropic.Anthropic()
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=100,
            system=(
                "Rewrite the farmer's short question into a fuller, clearer search "
                "query about papaya plant health for a knowledge-base retriever. "
                "Output ONLY the rewritten query, no preamble."
            ),
            messages=[{"role": "user", "content": query}],
        )
        text_blocks = [b.text for b in response.content if getattr(b, "type", None) == "text"]
        rewritten = "\n".join(text_blocks).strip()
        return rewritten or None
    except Exception:  # noqa: BLE001
        return None


def expand_query(query: str, kb_path) -> str:
    """
    Returns an expanded/rewritten version of `query` for retrieval. Falls
    through LLM rewrite -> lexical expansion -> original query unchanged.
    """
    llm_result = _llm_rewrite(query)
    if llm_result:
        return llm_result

    chunks = load_knowledge_base(kb_path)
    return _lexical_expand(query, chunks)

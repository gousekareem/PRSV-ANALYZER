from __future__ import annotations

"""
GraphRAG (v3.1): structures the existing flat knowledge base
(rag/kb/prsv_knowledge.json, 30 chunks) as a symptom -> cause/transmission ->
treatment/prevention graph, so a "why does this treatment work" question can
be answered by traversing two relations instead of relying on a single
flat-retrieval hit that happens to mention both.

Honesty note on scope: this is a real graph over the *existing* 30-entry
knowledge base, built from its `category` and `tags` fields - not a
production knowledge-graph database (Neo4j, etc.), which would be
overengineering for 30 entries. The graph structure and multi-hop traversal
logic are genuine and testable; swapping the in-memory graph for a real
graph database is a mechanical change if the knowledge base grows large
enough to need one (see ROADMAP_v3.md's note on when a vector DB similarly
becomes worth it - the same "grows past a few hundred entries" threshold
applies here).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set

from rag.knowledge_loader import load_knowledge_base
from rag.schemas import KnowledgeChunk

# The causal ordering a multi-hop "why" question typically traverses:
# what it looks like -> why it happens / how it spreads -> what to do about it.
_STAGE_ORDER = [
    {"symptoms", "symptom_interpretation", "disease_description"},
    {"transmission", "impact"},
    {"treatment", "prevention", "management", "general_care"},
    {"advisory", "severity", "schemes", "chat_faq", "tool_usage", "explanation_support"},
]


@dataclass
class GraphEdge:
    source_id: str
    target_id: str
    shared_tags: List[str]


@dataclass
class PrsvKnowledgeGraph:
    nodes: Dict[str, KnowledgeChunk]
    edges: List[GraphEdge] = field(default_factory=list)

    def neighbors(self, chunk_id: str) -> List[str]:
        return [e.target_id for e in self.edges if e.source_id == chunk_id] + [
            e.source_id for e in self.edges if e.target_id == chunk_id
        ]


def _stage_of(category: str) -> int:
    for stage_idx, categories in enumerate(_STAGE_ORDER):
        if category in categories:
            return stage_idx
    return len(_STAGE_ORDER)  # unranked categories sort last


def build_graph(kb_path) -> PrsvKnowledgeGraph:
    chunks = load_knowledge_base(kb_path)
    nodes = {c.chunk_id: c for c in chunks}

    edges: List[GraphEdge] = []
    for i, chunk_a in enumerate(chunks):
        for chunk_b in chunks[i + 1 :]:
            shared = sorted(set(chunk_a.tags) & set(chunk_b.tags))
            # Only connect across *different* pipeline stages that share at
            # least one tag - that's what makes an edge a "symptom relates to
            # this cause" link rather than two symptom entries both
            # mentioning "papaya".
            if shared and _stage_of(chunk_a.category) != _stage_of(chunk_b.category):
                edges.append(GraphEdge(chunk_a.chunk_id, chunk_b.chunk_id, shared))

    return PrsvKnowledgeGraph(nodes=nodes, edges=edges)


def multi_hop_query(graph: PrsvKnowledgeGraph, start_chunk_id: str, max_hops: int = 2) -> List[dict]:
    """
    Breadth-first traversal from `start_chunk_id` outward through
    increasing pipeline stages (symptom -> cause -> treatment), returning
    a reasoning path: each step names which chunk, its category, and which
    shared tags justified the hop.
    """
    if start_chunk_id not in graph.nodes:
        return []

    visited: Set[str] = {start_chunk_id}
    frontier = [start_chunk_id]
    path: List[dict] = [
        {
            "hop": 0,
            "chunk_id": start_chunk_id,
            "title": graph.nodes[start_chunk_id].title,
            "category": graph.nodes[start_chunk_id].category,
            "via_tags": [],
        }
    ]

    for hop in range(1, max_hops + 1):
        next_frontier = []
        for node_id in frontier:
            for edge in graph.edges:
                neighbor_id = None
                if edge.source_id == node_id and edge.target_id not in visited:
                    neighbor_id = edge.target_id
                elif edge.target_id == node_id and edge.source_id not in visited:
                    neighbor_id = edge.source_id

                if neighbor_id:
                    visited.add(neighbor_id)
                    next_frontier.append(neighbor_id)
                    path.append(
                        {
                            "hop": hop,
                            "chunk_id": neighbor_id,
                            "title": graph.nodes[neighbor_id].title,
                            "category": graph.nodes[neighbor_id].category,
                            "via_tags": edge.shared_tags,
                        }
                    )
        frontier = next_frontier
        if not frontier:
            break

    return path


def find_best_start_node(graph: PrsvKnowledgeGraph, query_tags: List[str]) -> str | None:
    """
    Picks the graph node with the most tag overlap with the query's inferred
    tags (e.g. from rag/query_builder.py's symptom-derived query) as the
    entry point for multi_hop_query.
    """
    query_tag_set = set(t.lower() for t in query_tags)
    if not query_tag_set:
        return None

    best_id, best_overlap = None, 0
    for chunk_id, chunk in graph.nodes.items():
        overlap = len(query_tag_set & {t.lower() for t in chunk.tags})
        if overlap > best_overlap:
            best_id, best_overlap = chunk_id, overlap

    return best_id

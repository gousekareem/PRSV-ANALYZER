from app.config import settings
from rag.agent_orchestrator import OrchestratorAction, route
from rag.corrective_rag import grade_and_decide
from rag.knowledge_graph import build_graph, find_best_start_node, multi_hop_query
from rag.query_expansion import expand_query
from rag.reranker import rerank
from rag.schemas import RetrievedChunk


def _chunk(chunk_id: str, text: str, category: str = "symptoms", tags=None) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        title=chunk_id,
        category=category,
        audience="farmer",
        text=text,
        tags=tags or [],
        similarity_score=1.0,
    )


def test_rerank_returns_same_or_fewer_items() -> None:
    candidates = [
        _chunk("c1", "PRSV causes yellow mosaic patterns."),
        _chunk("c2", "Weather forecast for tomorrow is sunny."),
        _chunk("c3", "Treatment includes removing infected plants."),
    ]
    result = rerank("What are PRSV symptoms?", candidates, top_k=2)
    assert len(result) <= 2
    assert all(isinstance(r, RetrievedChunk) for r in result)


def test_rerank_empty_candidates() -> None:
    assert rerank("query", []) == []


def test_expand_query_never_raises() -> None:
    expanded = expand_query("wat 2 do abt spots", settings.kb_path)
    assert isinstance(expanded, str)
    assert len(expanded) >= len("wat 2 do abt spots")


def test_corrective_rag_fallback_on_no_chunks() -> None:
    decision = grade_and_decide("PRSV symptoms", [])
    assert decision.action == "fallback"


def test_corrective_rag_generate_on_relevant_chunks() -> None:
    chunks = [
        _chunk("c1", "PRSV causes yellow mosaic patterns and ring spots on papaya leaves."),
        _chunk("c2", "Ring spot symptoms include leaf distortion and mottling."),
    ]
    decision = grade_and_decide("What are PRSV ring spot symptoms?", chunks)
    assert decision.action == "generate"
    assert len(decision.relevant_chunks) > 0


def test_knowledge_graph_builds_from_real_kb() -> None:
    graph = build_graph(settings.kb_path)
    assert len(graph.nodes) > 0
    # Real KB should have at least some cross-category tag-sharing edges.
    assert isinstance(graph.edges, list)


def test_knowledge_graph_multi_hop_query() -> None:
    graph = build_graph(settings.kb_path)
    start_id = find_best_start_node(graph, ["prsv", "symptoms"])
    if start_id:
        path = multi_hop_query(graph, start_id, max_hops=2)
        assert len(path) >= 1
        assert path[0]["chunk_id"] == start_id


def test_agent_orchestrator_routes_photo_mention() -> None:
    decision = route("here is a picture of my plant", has_attached_photo=True)
    assert decision.action == OrchestratorAction.ANALYZE_PHOTO


def test_agent_orchestrator_routes_knowledge_question() -> None:
    decision = route("how does papaya ring spot virus spread through a field", has_attached_photo=False)
    assert decision.action == OrchestratorAction.RETRIEVE_KNOWLEDGE


def test_agent_orchestrator_routes_greeting() -> None:
    decision = route("hi", has_attached_photo=False)
    assert decision.action == OrchestratorAction.DIRECT_ANSWER

from app.config import settings
from rag.hybrid_retriever import HybridRetriever
from rag.knowledge_loader import load_knowledge_base


def test_hybrid_retriever_returns_results_regardless_of_backend() -> None:
    """
    Whether or not sentence-transformers/faiss/rank_bm25 are installed in the
    test environment, HybridRetriever must always return relevant results by
    degrading through its fallback chain (hybrid -> bm25/dense-only ->
    tfidf). This is the property that matters for CI on a minimal install.
    """
    chunks = load_knowledge_base(settings.kb_path)
    retriever = HybridRetriever(chunks)

    assert retriever.active_mode in {"hybrid_rrf", "bm25_only", "dense_only", "tfidf_fallback"}

    results = retriever.retrieve("papaya leaf yellow mosaic symptoms", top_k=3)

    assert len(results) > 0
    assert all(r.text for r in results)


def test_hybrid_retriever_respects_top_k() -> None:
    chunks = load_knowledge_base(settings.kb_path)
    retriever = HybridRetriever(chunks)

    results = retriever.retrieve("severity treatment prevention", top_k=2)

    assert len(results) <= 2

from __future__ import annotations

from typing import Dict

from app.config import Settings
from app.schemas import ExplanationTrace, RetrievedEvidence
from app.services.cache_service import CacheService, make_cache_key
from rag.generator import (
    generate_advisory_notes,
    generate_farmer_friendly_explanation,
    generate_technical_explanation,
)
from rag.hybrid_retriever import HybridRetriever
from rag.knowledge_loader import load_knowledge_base
from rag.llm_generator import generate_grounded_explanation
from rag.query_builder import build_key_findings, build_observation_query
from rag.retriever import LocalTfidfRetriever


class RagService:
    """
    PRSV RAG service (v3.0).

    Retrieval: uses HybridRetriever (BM25 + dense embeddings fused via
    Reciprocal Rank Fusion) when settings.rag_use_hybrid_retrieval is True
    and the optional packages are installed; otherwise falls back to the
    original TF-IDF retriever, so behavior on a minimal install is identical
    to v2.9.

    Generation: attempts a real, grounded Claude API call
    (rag/llm_generator.py) when settings.rag_use_llm_generation is True and
    ANTHROPIC_API_KEY is configured. On any failure (no key, package missing,
    API error) it falls back to the original deterministic template
    generator (rag/generator.py), which always works and is what the
    technical/advisory-notes text used elsewhere in the pipeline still uses
    for consistency.

    Caching: repeated observation queries (e.g. many farmers hitting very
    similar feature ranges) are cached via CacheService to avoid redundant
    retrieval + generation work.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.chunks = load_knowledge_base(settings.kb_path)

        if settings.rag_use_hybrid_retrieval:
            self.retriever = HybridRetriever(self.chunks)
        else:
            self.retriever = LocalTfidfRetriever(self.chunks)

        self.cache = CacheService(settings)

    @property
    def retrieval_mode(self) -> str:
        if isinstance(self.retriever, HybridRetriever):
            return self.retriever.active_mode
        return "tfidf_fallback"

    def _retrieve(self, query: str):
        if isinstance(self.retriever, HybridRetriever):
            return self.retriever.retrieve(
                query=query,
                top_k=self.settings.rag_top_k,
                candidate_pool=self.settings.rag_candidate_pool_size,
            )
        return self.retriever.retrieve(query=query, top_k=self.settings.rag_top_k)

    def build_explanation_trace(
        self,
        prediction: str,
        confidence: float,
        severity_label: str,
        severity_score: float,
        feature_values: Dict[str, float],
        symptom_findings: Dict[str, float],
        segmentation_success: bool,
    ) -> ExplanationTrace:
        observation_query = build_observation_query(
            prediction=prediction,
            severity_label=severity_label,
            feature_values=feature_values,
            severity_score=severity_score,
            symptom_findings=symptom_findings,
        )

        cache_key = make_cache_key("rag_trace", observation_query, self.retrieval_mode)
        cached = self.cache.get_json(cache_key)
        if cached is not None:
            return ExplanationTrace(**cached)

        retrieved_chunks = self._retrieve(observation_query)

        retrieved_evidence = [
            RetrievedEvidence(
                chunk_id=chunk.chunk_id,
                title=chunk.title,
                text=chunk.text,
                similarity_score=chunk.similarity_score,
            )
            for chunk in retrieved_chunks
        ]

        key_findings = build_key_findings(
            prediction=prediction,
            confidence=confidence,
            severity_label=severity_label,
            severity_score=severity_score,
            feature_values=feature_values,
            symptom_findings=symptom_findings,
            segmentation_success=segmentation_success,
        )

        technical_explanation = generate_technical_explanation(
            prediction=prediction,
            confidence=confidence,
            severity_label=severity_label,
            severity_score=severity_score,
            feature_values=feature_values,
            retrieved_chunks=retrieved_chunks,
        )

        farmer_friendly_explanation = None
        if self.settings.rag_use_llm_generation:
            farmer_friendly_explanation = generate_grounded_explanation(
                prediction=prediction,
                confidence=confidence,
                severity_label=severity_label,
                severity_score=severity_score,
                retrieved_chunks=retrieved_chunks,
                audience="farmer",
            )

        if not farmer_friendly_explanation:
            farmer_friendly_explanation = generate_farmer_friendly_explanation(
                prediction=prediction,
                severity_label=severity_label,
                retrieved_chunks=retrieved_chunks,
            )

        advisory_notes = generate_advisory_notes(
            prediction=prediction,
            severity_label=severity_label,
        )

        trace = ExplanationTrace(
            observation_query=observation_query,
            key_findings=key_findings,
            retrieved_evidence=retrieved_evidence,
            technical_explanation=technical_explanation,
            farmer_friendly_explanation=farmer_friendly_explanation,
            advisory_notes=advisory_notes,
        )

        self.cache.set_json(cache_key, trace.model_dump())
        return trace

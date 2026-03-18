from __future__ import annotations

from typing import Dict

from app.config import Settings
from app.schemas import ExplanationTrace, RetrievedEvidence
from rag.generator import (
    generate_advisory_notes,
    generate_farmer_friendly_explanation,
    generate_technical_explanation,
)
from rag.knowledge_loader import load_knowledge_base
from rag.query_builder import build_key_findings, build_observation_query
from rag.retriever import LocalTfidfRetriever


class RagService:
    """
    Real local-first PRSV RAG service.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.chunks = load_knowledge_base(settings.kb_path)
        self.retriever = LocalTfidfRetriever(self.chunks)

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

        retrieved_chunks = self.retriever.retrieve(
            query=observation_query,
            top_k=self.settings.rag_top_k,
        )

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

        farmer_friendly_explanation = generate_farmer_friendly_explanation(
            prediction=prediction,
            severity_label=severity_label,
            retrieved_chunks=retrieved_chunks,
        )

        advisory_notes = generate_advisory_notes(
            prediction=prediction,
            severity_label=severity_label,
        )

        return ExplanationTrace(
            observation_query=observation_query,
            key_findings=key_findings,
            retrieved_evidence=retrieved_evidence,
            technical_explanation=technical_explanation,
            farmer_friendly_explanation=farmer_friendly_explanation,
            advisory_notes=advisory_notes,
        )
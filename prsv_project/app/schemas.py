from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict


class HealthStatus(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    app_name: str
    app_version: str
    demo_dataset_available: bool
    model_available: bool
    kb_available: bool


class RetrievedEvidence(BaseModel):
    chunk_id: str
    title: str
    text: str
    similarity_score: float


class ExplanationTrace(BaseModel):
    observation_query: str
    key_findings: List[str]
    retrieved_evidence: List[RetrievedEvidence]
    technical_explanation: str
    farmer_friendly_explanation: str
    advisory_notes: List[str]


class ImageResult(BaseModel):
    image_id: str
    filename: str
    prediction: str
    confidence: float
    infection_percentage: float
    severity_score: float
    severity_label: str
    feature_values: Dict[str, float]
    output_paths: Dict[str, str]
    explanation_trace: ExplanationTrace


class BatchResult(BaseModel):
    run_id: str
    total_images: int
    processed_images: int
    failed_images: int
    healthy_count: int
    diseased_count: int
    average_confidence: float
    average_infection_percentage: float
    severity_distribution: Dict[str, int]
    results: List[ImageResult]
    failures: List[Dict[str, Any]]


class ErrorResponse(BaseModel):
    detail: str
    context: Optional[Dict[str, Any]] = None
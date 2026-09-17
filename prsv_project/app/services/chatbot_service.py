from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from app.config import Settings
from app.services.analysis_service import AnalysisService
from app.services.batch_service import BatchService
from app.services.run_manager import RunManager
from app.services.translation_service import (
    DEFAULT_LANGUAGE,
    get_phrase,
    is_supported_language,
    translate_text,
)
from app.services.cache_service import CacheService, make_cache_key
from app.utils.display_labels import prediction_display_label
from nlp.intent_classifier import Intent, classify_intent
from nlp.symptom_ner import extract_symptom_findings
from nlp.urgency_detection import assess_urgency
from rag.knowledge_loader import load_knowledge_base
from rag.hybrid_retriever import HybridRetriever
from rag.retriever import LocalTfidfRetriever

_GREETING_PATTERN = re.compile(
    r"^\s*(hi|hello|hey|namaste|namaskar|vanakkam|help)\b", re.IGNORECASE
)


class ChatbotService:
    """
    The chatbot's "brain": retrieval-grounded Q&A over the same PRSV
    knowledge base the main analysis pipeline uses, plus the ability to run
    a photo straight through the existing analysis pipeline and explain the
    result conversationally. No external LLM call - answers are composed
    from retrieved knowledge chunks, consistent with how the rest of this
    app already works (kept deliberately consistent rather than mixing a
    templated system with an LLM-backed chat feature that behaves
    differently).
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.chunks = load_knowledge_base(settings.kb_path)
        # v3.0: hybrid BM25 + dense retrieval (auto-degrades to TF-IDF if the
        # optional embedding/BM25 packages aren't installed - see
        # rag/hybrid_retriever.py) instead of the original TF-IDF-only path.
        if settings.rag_use_hybrid_retrieval:
            self.retriever = HybridRetriever(self.chunks)
        else:
            self.retriever = LocalTfidfRetriever(self.chunks)
        self.cache = CacheService(settings)
        self.run_manager = RunManager(settings)
        self.analysis_service = AnalysisService(settings, self.run_manager)
        self.batch_service = BatchService(settings, self.run_manager, self.analysis_service)

    def _to_english(self, text: str, source_lang: str) -> tuple[str, bool]:
        if source_lang == "en":
            return text, True
        result = translate_text(text, target_lang="en", source_lang=source_lang)
        return result.text, result.used_live_translation

    def _from_english(self, text: str, target_lang: str) -> tuple[str, bool]:
        if target_lang == "en":
            return text, True
        result = translate_text(text, target_lang=target_lang, source_lang="en")
        return result.text, result.used_live_translation

    def _compose_answer_from_chunks(self, query_en: str) -> str:
        # Cache repeated FAQ-style questions (many farmers asking near-identical
        # things) to avoid re-running retrieval every time. Cache key is scoped
        # to the normalized English query only - translation happens outside
        # this method, so the cached English answer is reused regardless of
        # which language the farmer originally typed in.
        cache_key = make_cache_key("chatbot_answer", query_en.strip().lower())
        cached = self.cache.get_json(cache_key)
        if cached is not None:
            return cached

        retrieved = self.retriever.retrieve(query=query_en, top_k=3)
        farmer_chunks = [c for c in retrieved if c.audience == "farmer"] or retrieved

        if not farmer_chunks:
            answer = (
                "I don't have specific information on that yet. I can help with papaya Ring Spot "
                "Virus symptoms, prevention, treatment, and checking a leaf photo."
            )
            return answer

        # Compose from the best-matching chunk plus a second for extra context,
        # rather than dumping all retrieved text verbatim.
        primary = farmer_chunks[0].text
        answer = primary
        if len(farmer_chunks) > 1 and farmer_chunks[1].text not in primary:
            answer += " " + farmer_chunks[1].text

        self.cache.set_json(cache_key, answer)
        return answer

    def handle_message(self, text: str, language: str) -> dict:
        language = language if is_supported_language(language) else DEFAULT_LANGUAGE
        text = (text or "").strip()

        # v3.1: real intent classification (embedding-based zero-shot,
        # degrading to keyword heuristic - see nlp/intent_classifier.py)
        # replacing the old regex-only greeting check, plus urgency
        # detection and structured symptom extraction surfaced in the
        # response metadata for any caller that wants to act on them
        # (e.g. routing urgent messages to a human reviewer).
        intent = classify_intent(text) if text else Intent.QUESTION
        urgency = assess_urgency(text)
        symptom_findings = extract_symptom_findings(text) if text else None

        if not text:
            reply_en = "Please type a question, or attach a leaf photo to check it."
        elif intent == Intent.GREETING or _GREETING_PATTERN.match(text):
            reply_translated = get_phrase("greeting", language)
            return {
                "reply": reply_translated,
                "reply_english": get_phrase("greeting", "en"),
                "language": language,
                "used_live_translation": True,  # phrasebook is pre-translated, always "available"
                "source": "phrasebook",
                "intent": intent.value,
                "urgency": urgency.level,
            }
        else:
            query_en, translation_ok_in = self._to_english(text, language)
            reply_en = self._compose_answer_from_chunks(query_en)
            if urgency.level == "urgent":
                reply_en = (
                    "This sounds serious, so please prioritize contacting a local agricultural "
                    "extension officer as soon as possible. " + reply_en
                )

        reply_translated, translation_ok_out = self._from_english(reply_en, language)

        note = "" if translation_ok_out else (" " + get_phrase("translation_unavailable", language))

        return {
            "reply": reply_translated + note,
            "reply_english": reply_en,
            "language": language,
            "used_live_translation": translation_ok_out,
            "source": "knowledge_base",
            "intent": intent.value,
            "urgency": urgency.level,
            "symptom_findings": {
                "symptoms": symptom_findings.symptoms,
                "plant_parts": symptom_findings.plant_parts,
            }
            if symptom_findings
            else None,
        }

    def handle_photo(self, image_path: Path, language: str) -> dict:
        language = language if is_supported_language(language) else DEFAULT_LANGUAGE

        batch_result = self.batch_service.analyze_images([image_path])
        if not batch_result.results:
            error_translated, _ = self._from_english(
                "I couldn't process that photo. Please try a clearer, well-lit picture of the leaf.",
                language,
            )
            return {"reply": error_translated, "language": language, "source": "photo_analysis_error", "error": True}

        result = batch_result.results[0]

        is_healthy = result.prediction.strip().lower() == "healthy"
        intro_key = "result_healthy" if is_healthy else "result_diseased"
        intro_en = get_phrase(intro_key, "en")

        display_prediction = prediction_display_label(result.prediction)
        summary_en = (
            f"{intro_en} Prediction: {display_prediction} ({result.confidence * 100:.0f}% confidence). "
            f"Severity: {result.severity_label} ({result.severity_score:.0f}%). "
            f"{result.explanation_trace.farmer_friendly_explanation}"
        )
        if result.explanation_trace.advisory_notes:
            summary_en += " Next steps: " + " ".join(result.explanation_trace.advisory_notes[:3])

        reply_translated, translation_ok = self._from_english(summary_en, language)
        note = "" if translation_ok else (" " + get_phrase("translation_unavailable", language))

        return {
            "reply": reply_translated + note,
            "reply_english": summary_en,
            "language": language,
            "used_live_translation": translation_ok,
            "prediction": result.prediction,
            "confidence": result.confidence,
            "severity_label": result.severity_label,
            "severity_score": result.severity_score,
            "image_id": result.image_id,
            "run_id": batch_result.run_id,
            "source": "photo_analysis",
        }

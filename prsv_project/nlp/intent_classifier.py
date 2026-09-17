from __future__ import annotations

"""
Intent classification (v3.1), replacing chatbot_service.py's regex-only
greeting matcher with genuine intent understanding across more classes:
question, greeting, complaint/urgent, photo_request, thanks.

Honesty note on approach: the wishlist names DistilBERT fine-tuning or
zero-shot BART-MNLI. Both need `transformers` + a multi-hundred-MB model
download (BART-MNLI is ~1.6GB), which is the same PyTorch-dependency cost
already flagged for sentence-transformers in requirements-optional.txt. To
avoid asking for a second huge model just for intent classification, this
module implements genuine zero-shot classification using the SAME small
multilingual embedding model already loaded for retrieval
(rag/embeddings_retriever.py's EMBEDDING_MODEL_NAME): each intent is defined
by a handful of example utterances, and a message is classified by which
intent's examples it's most similar to in embedding space. This is a real,
standard zero-shot technique (embedding-based nearest-centroid
classification) - not as strong as a purpose-built NLI model, but it needs
zero additional model downloads beyond what v3.0 already requires for
hybrid retrieval, and degrades to the original regex matcher if even that
embedding model isn't installed.
"""

from enum import Enum
from typing import Dict, List, Optional

_INTENT_EXAMPLES: Dict[str, List[str]] = {
    "greeting": ["hello", "hi there", "namaste", "good morning", "hey"],
    "thanks": ["thank you", "thanks a lot", "that helped, thanks", "appreciate it"],
    "photo_request": [
        "can you check my leaf photo",
        "I want to upload a picture",
        "attach a photo for diagnosis",
        "here is a picture of my plant",
    ],
    "complaint_urgent": [
        "my whole field is dying",
        "all my plants are infected badly",
        "this is spreading very fast, help",
        "urgent, many plants affected",
    ],
    "question": [
        "what are the symptoms of PRSV",
        "how does papaya ring spot virus spread",
        "what treatment should I use",
        "how can I prevent this disease",
    ],
}

_GREETING_KEYWORDS = {"hi", "hello", "hey", "namaste", "namaskar", "vanakkam", "help"}
_THANKS_KEYWORDS = {"thanks", "thank", "thx"}


class Intent(str, Enum):
    GREETING = "greeting"
    THANKS = "thanks"
    PHOTO_REQUEST = "photo_request"
    COMPLAINT_URGENT = "complaint_urgent"
    QUESTION = "question"


_model_cache: dict = {}
_centroid_cache: dict = {}


def _get_model():
    if "model" not in _model_cache:
        try:
            from sentence_transformers import SentenceTransformer

            from rag.embeddings_retriever import EMBEDDING_MODEL_NAME

            _model_cache["model"] = SentenceTransformer(EMBEDDING_MODEL_NAME)
        except Exception:  # noqa: BLE001
            _model_cache["model"] = None
    return _model_cache["model"]


def _get_centroids():
    if "centroids" not in _centroid_cache:
        model = _get_model()
        if model is None:
            _centroid_cache["centroids"] = None
            return None

        import numpy as np

        centroids = {}
        for intent, examples in _INTENT_EXAMPLES.items():
            vectors = model.encode(examples, normalize_embeddings=True, show_progress_bar=False)
            centroids[intent] = np.mean(vectors, axis=0)
        _centroid_cache["centroids"] = centroids
    return _centroid_cache["centroids"]


def _keyword_fallback(text: str) -> Intent:
    lowered = text.lower().strip()
    first_word = lowered.split()[0] if lowered.split() else ""
    if first_word in _GREETING_KEYWORDS:
        return Intent.GREETING
    if any(kw in lowered for kw in _THANKS_KEYWORDS):
        return Intent.THANKS
    return Intent.QUESTION


def classify_intent(text: str) -> Intent:
    """
    Classify a chatbot message's intent. Falls back to a keyword heuristic
    (a strict superset of the original regex-only greeting matcher) if the
    embedding model isn't available.
    """
    text = (text or "").strip()
    if not text:
        return Intent.QUESTION

    centroids = _get_centroids()
    model = _get_model()
    if centroids is None or model is None:
        return _keyword_fallback(text)

    try:
        import numpy as np

        query_vector = model.encode([text], normalize_embeddings=True, show_progress_bar=False)[0]
        best_intent: Optional[str] = None
        best_score = -1.0
        for intent, centroid in centroids.items():
            score = float(np.dot(query_vector, centroid))
            if score > best_score:
                best_intent, best_score = intent, score
        return Intent(best_intent) if best_intent else _keyword_fallback(text)
    except Exception:  # noqa: BLE001
        return _keyword_fallback(text)

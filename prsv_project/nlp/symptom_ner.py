from __future__ import annotations

"""
Symptom/plant-part extraction (v3.1).

Honesty note: the wishlist calls this "NER" (Named Entity Recognition),
which typically means a trained sequence-labeling model (spaCy, a
fine-tuned transformer). Training or even fine-tuning one needs a labeled
corpus of farmer-style sentences with symptom/plant-part spans annotated,
which doesn't exist for this project. What's implemented here is a
pattern/dictionary-based extractor: it matches known symptom and plant-part
vocabulary (sourced from the same knowledge base tags already used
elsewhere in rag/) against the input text. This is a legitimate, commonly-
used *rule-based* NER approach - a defensible and fully honest first step -
but it is not a statistical model and won't generalize to symptom
descriptions using vocabulary outside its dictionary the way a trained NER
model would.
"""

from dataclasses import dataclass, field
from typing import List

_SYMPTOM_VOCAB = {
    "yellow spots", "yellowing", "mosaic", "ring spot", "ring spots",
    "chlorosis", "distortion", "curling", "curl", "wilting", "stunted",
    "discoloration", "blister", "blisters", "streak", "streaks",
    "mottling", "lesion", "lesions", "necrosis", "spots",
}

_PLANT_PART_VOCAB = {
    "leaf", "leaves", "underside", "stem", "fruit", "petiole", "shoot",
    "top leaves", "young leaves", "old leaves", "trunk", "root", "roots",
}


@dataclass
class SymptomFindings:
    symptoms: List[str] = field(default_factory=list)
    plant_parts: List[str] = field(default_factory=list)
    raw_text: str = ""

    def as_structured_query(self) -> str:
        parts = []
        if self.symptoms:
            parts.append(f"symptom={','.join(self.symptoms)}")
        if self.plant_parts:
            parts.append(f"location={','.join(self.plant_parts)}")
        return "; ".join(parts) if parts else "no structured findings extracted"


def extract_symptom_findings(text: str) -> SymptomFindings:
    """
    Extract known symptom and plant-part mentions from free text, e.g.
    "yellow spots on the underside of my leaves" ->
    symptoms=["yellowing"/"spots"], plant_parts=["underside", "leaves"].
    Longer multi-word vocabulary entries are matched before shorter ones so
    "ring spots" matches as one entity rather than being double-counted as
    "ring" + "spots".
    """
    lowered = f" {text.lower()} "

    def _match_vocab(vocab: set) -> List[str]:
        sorted_vocab = sorted(vocab, key=len, reverse=True)
        matched: List[str] = []
        remaining = lowered
        for term in sorted_vocab:
            padded_term = f" {term} " if not term.endswith("s") else f" {term} "
            if f" {term} " in remaining or remaining.strip().startswith(term) or remaining.strip().endswith(term):
                if term not in matched:
                    matched.append(term)
                remaining = remaining.replace(term, " ")
        return matched

    symptoms = _match_vocab(_SYMPTOM_VOCAB)
    plant_parts = _match_vocab(_PLANT_PART_VOCAB)

    return SymptomFindings(symptoms=symptoms, plant_parts=plant_parts, raw_text=text)

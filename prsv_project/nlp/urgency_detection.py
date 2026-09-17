from __future__ import annotations

"""
Urgency / distress detection (v3.1): flags a chatbot message for a
different, more urgent response tier - e.g. "my whole field is dying"
should not get the same reply pacing as "what is PRSV".

Honesty note: this is a keyword + heuristic scorer, not a trained sentiment
model (which would need labeled farmer-message data this project doesn't
have). It's a legitimate, transparent first pass - each signal is
inspectable - but will miss distress phrased outside its keyword list the
way a trained classifier would generalize to.
"""

from dataclasses import dataclass
from typing import List

_URGENT_KEYWORDS = {
    "whole field", "all my plants", "everything is dying", "spreading fast",
    "urgent", "emergency", "losing everything", "many plants", "entire crop",
    "help me", "desperate", "ruined",
}

_DISTRESS_PUNCTUATION_SIGNAL = ("!!!", "???", "please help")

_MILD_NEGATIVE_KEYWORDS = {"worried", "scared", "concerned", "not sure", "confused"}


@dataclass
class UrgencyAssessment:
    level: str  # "normal", "elevated", "urgent"
    score: float
    matched_signals: List[str]


def assess_urgency(text: str) -> UrgencyAssessment:
    lowered = (text or "").lower()
    matched: List[str] = []
    score = 0.0

    for keyword in _URGENT_KEYWORDS:
        if keyword in lowered:
            matched.append(keyword)
            score += 1.0

    for signal in _DISTRESS_PUNCTUATION_SIGNAL:
        if signal in lowered:
            matched.append(signal)
            score += 0.5

    for keyword in _MILD_NEGATIVE_KEYWORDS:
        if keyword in lowered:
            matched.append(keyword)
            score += 0.3

    if score >= 1.5:
        level = "urgent"
    elif score >= 0.5:
        level = "elevated"
    else:
        level = "normal"

    return UrgencyAssessment(level=level, score=round(score, 2), matched_signals=matched)

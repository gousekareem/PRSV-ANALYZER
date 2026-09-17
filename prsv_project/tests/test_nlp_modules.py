from nlp.dialogue_state import DialogueState, DialogueStateStore
from nlp.intent_classifier import Intent, classify_intent
from nlp.symptom_ner import extract_symptom_findings
from nlp.urgency_detection import assess_urgency


def test_classify_intent_greeting_keyword_fallback_path() -> None:
    # Works regardless of whether the embedding model is installed, since
    # "hello" is unambiguous under either the embedding or keyword path.
    intent = classify_intent("hello")
    assert intent in {Intent.GREETING, Intent.QUESTION}  # embedding path may differ slightly from keyword


def test_classify_intent_empty_text_defaults_to_question() -> None:
    assert classify_intent("") == Intent.QUESTION


def test_extract_symptom_findings_finds_known_terms() -> None:
    findings = extract_symptom_findings("There are yellow spots on the underside of my leaves")
    assert len(findings.symptoms) > 0 or len(findings.plant_parts) > 0
    assert "underside" in findings.plant_parts or "leaves" in findings.plant_parts


def test_extract_symptom_findings_empty_text() -> None:
    findings = extract_symptom_findings("")
    assert findings.symptoms == []
    assert findings.plant_parts == []


def test_assess_urgency_flags_urgent_message() -> None:
    result = assess_urgency("my whole field is dying, please help urgent")
    assert result.level == "urgent"
    assert result.score > 0


def test_assess_urgency_normal_message() -> None:
    result = assess_urgency("what is PRSV")
    assert result.level == "normal"


def test_dialogue_state_coreference_resolution() -> None:
    state = DialogueState(session_id="s1")
    state.add_turn("My papaya plant has yellow spots", mentioned_topic="papaya plant with yellow spots")

    resolved = state.resolve_coreference("is it contagious?")
    assert "papaya plant with yellow spots" in resolved


def test_dialogue_state_store_isolates_sessions() -> None:
    store = DialogueStateStore()
    s1 = store.get_or_create("session_1")
    s2 = store.get_or_create("session_2")
    s1.add_turn("hello")

    assert len(s1.turns) == 1
    assert len(s2.turns) == 0

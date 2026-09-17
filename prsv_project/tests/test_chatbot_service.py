from app.config import settings
from app.services.chatbot_service import ChatbotService


def test_chatbot_greeting_returns_phrasebook_response() -> None:
    service = ChatbotService(settings)
    result = service.handle_message(text="hello", language="en")

    assert result["source"] == "phrasebook"
    assert "papaya" in result["reply"].lower() or "leaf" in result["reply"].lower()


def test_chatbot_answers_knowledge_question() -> None:
    service = ChatbotService(settings)
    result = service.handle_message(text="how does the virus spread", language="en")

    assert result["source"] == "knowledge_base"
    assert len(result["reply"]) > 0


def test_chatbot_falls_back_to_default_language_for_unsupported_code() -> None:
    service = ChatbotService(settings)
    result = service.handle_message(text="hello", language="zz")

    assert result["language"] == "en"

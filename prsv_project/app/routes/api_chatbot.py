from __future__ import annotations

from pydantic import BaseModel

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.config import settings
from app.services.chatbot_service import ChatbotService
from app.services.translation_service import list_supported_languages
from app.utils.file_utils import save_upload_file
from app.utils.validation_utils import (
    is_allowed_extension,
    validate_image_readable,
    validate_non_empty_file,
)

router = APIRouter(prefix="/api/chatbot", tags=["chatbot"])

_chatbot_service: ChatbotService | None = None


def _get_chatbot_service() -> ChatbotService:
    global _chatbot_service
    if _chatbot_service is None:
        _chatbot_service = ChatbotService(settings)
    return _chatbot_service


class ChatMessageRequest(BaseModel):
    text: str
    language: str = "en"


@router.get("/languages")
def get_languages() -> dict:
    return {"languages": list_supported_languages()}


@router.post("/message")
def post_message(payload: ChatMessageRequest) -> dict:
    if len(payload.text) > 2000:
        raise HTTPException(status_code=400, detail="Message is too long (max 2000 characters).")

    service = _get_chatbot_service()
    return service.handle_message(text=payload.text, language=payload.language)


@router.post("/analyze-image")
async def post_analyze_image(file: UploadFile = File(...), language: str = "en") -> dict:
    if not is_allowed_extension(file.filename or "", settings.allowed_extensions):
        raise HTTPException(status_code=400, detail="Unsupported file type.")

    saved_path = await save_upload_file(file, settings.upload_dir)

    if not validate_non_empty_file(saved_path):
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    if not validate_image_readable(saved_path):
        raise HTTPException(status_code=400, detail="Uploaded image is unreadable or corrupt.")

    service = _get_chatbot_service()
    return service.handle_photo(image_path=saved_path, language=language)

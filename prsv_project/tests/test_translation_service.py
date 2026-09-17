from app.services.translation_service import (
    get_phrase,
    is_supported_language,
    list_supported_languages,
)


def test_supported_languages_include_major_indian_languages() -> None:
    codes = {lang["code"] for lang in list_supported_languages()}
    for expected in {"en", "hi", "te", "ta", "kn", "ml", "bn", "mr", "gu", "pa"}:
        assert expected in codes


def test_is_supported_language() -> None:
    assert is_supported_language("hi") is True
    assert is_supported_language("xx") is False


def test_phrasebook_returns_native_script_for_hindi() -> None:
    phrase = get_phrase("greeting", "hi")
    assert phrase
    assert phrase != get_phrase("greeting", "en")


def test_phrasebook_falls_back_to_english_for_unsupported_language() -> None:
    phrase = get_phrase("greeting", "xx")
    assert phrase == get_phrase("greeting", "en")

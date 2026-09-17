from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

_logger = logging.getLogger("prsv.translation")

# code: (English name, native name, BCP-47 locale for Web Speech API)
SUPPORTED_LANGUAGES: dict[str, tuple[str, str, str]] = {
    "en": ("English", "English", "en-IN"),
    "hi": ("Hindi", "हिन्दी", "hi-IN"),
    "te": ("Telugu", "తెలుగు", "te-IN"),
    "ta": ("Tamil", "தமிழ்", "ta-IN"),
    "kn": ("Kannada", "ಕನ್ನಡ", "kn-IN"),
    "ml": ("Malayalam", "മലയാളം", "ml-IN"),
    "bn": ("Bengali", "বাংলা", "bn-IN"),
    "mr": ("Marathi", "मराठी", "mr-IN"),
    "gu": ("Gujarati", "ગુજરાતી", "gu-IN"),
    "pa": ("Punjabi", "ਪੰਜਾਬੀ", "pa-IN"),
}

DEFAULT_LANGUAGE = "en"


def list_supported_languages() -> list[dict[str, str]]:
    return [
        {"code": code, "name": name, "native_name": native, "speech_locale": locale}
        for code, (name, native, locale) in SUPPORTED_LANGUAGES.items()
    ]


def is_supported_language(code: str) -> bool:
    return code in SUPPORTED_LANGUAGES


@dataclass
class TranslationResult:
    text: str
    used_live_translation: bool


def translate_text(text: str, target_lang: str, source_lang: str = "en") -> TranslationResult:
    """
    Translate text using deep-translator's free Google Translate backend (no
    API key). This requires outbound internet access at runtime; if the
    request fails (offline, blocked network, rate-limited, etc.), the
    original text is returned unchanged with used_live_translation=False so
    the chatbot degrades gracefully instead of erroring out. Callers that
    need a response in the target language even when translation is
    unavailable should prefer the phrasebook (see below) for known intents.
    """
    if not text or target_lang == source_lang or target_lang not in SUPPORTED_LANGUAGES:
        return TranslationResult(text=text, used_live_translation=False)

    try:
        from deep_translator import GoogleTranslator

        translated = GoogleTranslator(source=source_lang, target=target_lang).translate(text)
        if translated:
            return TranslationResult(text=translated, used_live_translation=True)
    except Exception as exc:  # noqa: BLE001 - translation is best-effort, never fatal
        _logger.warning("Live translation unavailable (%s); falling back to English/phrasebook.", exc)

    return TranslationResult(text=text, used_live_translation=False)


# --- Offline phrasebook fallback -------------------------------------------
# Pre-translated canned phrases for the chatbot's most common moments, used
# when live translation is unavailable (no internet, blocked, rate-limited)
# so the widget still communicates *something* meaningful in the farmer's
# language rather than silently falling back to English. This is necessarily
# a small, curated set - free-form Q&A answers still depend on live
# translation when it's reachable.

PHRASEBOOK: dict[str, dict[str, str]] = {
    "greeting": {
        "en": "Hello! I can check a papaya leaf photo for Ring Spot Virus, or answer questions about it. You can type, speak, or attach a photo.",
        "hi": "नमस्ते! मैं पपीते के पत्ते की फोटो में रिंग स्पॉट वायरस की जांच कर सकता हूं, या इसके बारे में सवालों के जवाब दे सकता हूं। आप टाइप कर सकते हैं, बोल सकते हैं, या फोटो भेज सकते हैं।",
        "te": "నమస్కారం! నేను బొప్పాయి ఆకు ఫోటోలో రింగ్ స్పాట్ వైరస్‌ని తనిఖీ చేయగలను, లేదా దాని గురించి ప్రశ్నలకు సమాధానం ఇవ్వగలను. మీరు టైప్ చేయవచ్చు, మాట్లాడవచ్చు, లేదా ఫోటో పంపవచ్చు.",
        "ta": "வணக்கம்! பப்பாளி இலை புகைப்படத்தில் ரிங் ஸ்பாட் வைரஸை நான் சரிபார்க்க முடியும், அல்லது அதைப் பற்றிய கேள்விகளுக்கு பதிலளிக்க முடியும். நீங்கள் தட்டச்சு செய்யலாம், பேசலாம் அல்லது புகைப்படம் அனுப்பலாம்.",
        "kn": "ನಮಸ್ಕಾರ! ನಾನು ಪಪ್ಪಾಯಿ ಎಲೆಯ ಫೋಟೋದಲ್ಲಿ ರಿಂಗ್ ಸ್ಪಾಟ್ ವೈರಸ್ ಅನ್ನು ಪರಿಶೀಲಿಸಬಲ್ಲೆ, ಅಥವಾ ಅದರ ಬಗ್ಗೆ ಪ್ರಶ್ನೆಗಳಿಗೆ ಉತ್ತರಿಸಬಲ್ಲೆ. ನೀವು ಟೈಪ್ ಮಾಡಬಹುದು, ಮಾತನಾಡಬಹುದು ಅಥವಾ ಫೋಟೋ ಕಳುಹಿಸಬಹುದು.",
        "ml": "നമസ്കാരം! പപ്പായ ഇലയുടെ ഫോട്ടോയിൽ റിംഗ് സ്പോട്ട് വൈറസ് ഉണ്ടോ എന്ന് എനിക്ക് പരിശോധിക്കാം, അല്ലെങ്കിൽ അതിനെക്കുറിച്ചുള്ള ചോദ്യങ്ങൾക്ക് ഉത്തരം നൽകാം. നിങ്ങൾക്ക് ടൈപ്പ് ചെയ്യാം, സംസാരിക്കാം, അല്ലെങ്കിൽ ഫോട്ടോ അയക്കാം.",
        "bn": "নমস্কার! আমি পেঁপে পাতার ছবিতে রিং স্পট ভাইরাস পরীক্ষা করতে পারি, অথবা এই সম্পর্কে প্রশ্নের উত্তর দিতে পারি। আপনি টাইপ করতে পারেন, কথা বলতে পারেন, অথবা ছবি পাঠাতে পারেন।",
        "mr": "नमस्कार! मी पपईच्या पानाच्या फोटोमध्ये रिंग स्पॉट व्हायरस तपासू शकतो, किंवा त्याबद्दलच्या प्रश्नांची उत्तरे देऊ शकतो. तुम्ही टाइप करू शकता, बोलू शकता किंवा फोटो पाठवू शकता.",
        "gu": "નમસ્તે! હું પપૈયાના પાનના ફોટામાં રિંગ સ્પોટ વાયરસ તપાસી શકું છું, અથવા તેના વિશેના પ્રશ્નોના જવાબ આપી શકું છું. તમે ટાઇપ કરી શકો છો, બોલી શકો છો, અથવા ફોટો મોકલી શકો છો.",
        "pa": "ਸਤਿ ਸ੍ਰੀ ਅਕਾਲ! ਮੈਂ ਪਪੀਤੇ ਦੇ ਪੱਤੇ ਦੀ ਫੋਟੋ ਵਿੱਚ ਰਿੰਗ ਸਪਾਟ ਵਾਇਰਸ ਦੀ ਜਾਂਚ ਕਰ ਸਕਦਾ ਹਾਂ, ਜਾਂ ਇਸ ਬਾਰੇ ਸਵਾਲਾਂ ਦੇ ਜਵਾਬ ਦੇ ਸਕਦਾ ਹਾਂ। ਤੁਸੀਂ ਟਾਈਪ ਕਰ ਸਕਦੇ ਹੋ, ਬੋਲ ਸਕਦੇ ਹੋ, ਜਾਂ ਫੋਟੋ ਭੇਜ ਸਕਦੇ ਹੋ।",
    },
    "ask_for_photo": {
        "en": "Please attach a clear photo of the papaya leaf using the photo button.",
        "hi": "कृपया फोटो बटन का उपयोग करके पपीते के पत्ते की एक स्पष्ट फोटो जोड़ें।",
        "te": "దయచేసి ఫోటో బటన్ ఉపయోగించి బొప్పాయి ఆకు యొక్క స్పష్టమైన ఫోటోను జోడించండి.",
        "ta": "புகைப்பட பொத்தானைப் பயன்படுத்தி பப்பாளி இலையின் தெளிவான புகைப்படத்தை இணைக்கவும்.",
        "kn": "ದಯವಿಟ್ಟು ಫೋಟೋ ಬಟನ್ ಬಳಸಿ ಪಪ್ಪಾಯಿ ಎಲೆಯ ಸ್ಪಷ್ಟ ಫೋಟೋವನ್ನು ಲಗತ್ತಿಸಿ.",
        "ml": "ദയവായി ഫോട്ടോ ബട്ടൺ ഉപയോഗിച്ച് പപ്പായ ഇലയുടെ വ്യക്തമായ ഫോട്ടോ അറ്റാച്ച് ചെയ്യുക.",
        "bn": "অনুগ্রহ করে ফটো বোতাম ব্যবহার করে পেঁপে পাতার একটি স্পষ্ট ছবি সংযুক্ত করুন।",
        "mr": "कृपया फोटो बटण वापरून पपईच्या पानाचा स्पष्ट फोटो जोडा.",
        "gu": "કૃપા કરીને ફોટો બટનનો ઉપયોગ કરીને પપૈયાના પાનનો સ્પષ્ટ ફોટો જોડો.",
        "pa": "ਕਿਰਪਾ ਕਰਕੇ ਫੋਟੋ ਬਟਨ ਦੀ ਵਰਤੋਂ ਕਰਕੇ ਪਪੀਤੇ ਦੇ ਪੱਤੇ ਦੀ ਸਪਸ਼ਟ ਫੋਟੋ ਨੱਥੀ ਕਰੋ।",
    },
    "analyzing_photo": {
        "en": "Checking your photo, one moment...",
        "hi": "आपकी फोटो जांची जा रही है, कृपया प्रतीक्षा करें...",
        "te": "మీ ఫోటోను తనిఖీ చేస్తున్నాము, ఒక్క క్షణం...",
        "ta": "உங்கள் புகைப்படத்தை சரிபார்க்கிறோம், ஒரு நிமிடம்...",
        "kn": "ನಿಮ್ಮ ಫೋಟೋವನ್ನು ಪರಿಶೀಲಿಸಲಾಗುತ್ತಿದೆ, ಒಂದು ಕ್ಷಣ...",
        "ml": "നിങ്ങളുടെ ഫോട്ടോ പരിശോധിക്കുന്നു, ഒരു നിമിഷം...",
        "bn": "আপনার ছবি পরীক্ষা করা হচ্ছে, একটু অপেক্ষা করুন...",
        "mr": "तुमचा फोटो तपासला जात आहे, कृपया थांबा...",
        "gu": "તમારો ફોટો તપાસી રહ્યા છીએ, એક ક્ષણ...",
        "pa": "ਤੁਹਾਡੀ ਫੋਟੋ ਦੀ ਜਾਂਚ ਹੋ ਰਹੀ ਹੈ, ਇੱਕ ਪਲ...",
    },
    "translation_unavailable": {
        "en": "(Live translation is temporarily unavailable, showing English.)",
        "hi": "(लाइव अनुवाद अस्थायी रूप से अनुपलब्ध है, अंग्रेज़ी दिखाई जा रही है।)",
        "te": "(ప్రత్యక్ష అనువాదం తాత్కాలికంగా అందుబాటులో లేదు, ఇంగ్లీష్ చూపిస్తోంది.)",
        "ta": "(நேரடி மொழிபெயர்ப்பு தற்காலிகமாகக் கிடைக்கவில்லை, ஆங்கிலம் காட்டப்படுகிறது.)",
        "kn": "(ಲೈವ್ ಅನುವಾದ ತಾತ್ಕಾಲಿಕವಾಗಿ ಲಭ್ಯವಿಲ್ಲ, ಇಂಗ್ಲಿಷ್ ತೋರಿಸಲಾಗುತ್ತಿದೆ.)",
        "ml": "(തത്സമയ വിവർത്തനം താൽക്കാലികമായി ലഭ്യമല്ല, ഇംഗ്ലീഷ് കാണിക്കുന്നു.)",
        "bn": "(লাইভ অনুবাদ সাময়িকভাবে অনুপলব্ধ, ইংরেজি দেখানো হচ্ছে।)",
        "mr": "(थेट भाषांतर तात्पुरते अनुपलब्ध आहे, इंग्रजी दाखवत आहे.)",
        "gu": "(લાઇવ અનુવાદ કામચલાઉ રૂપે અનુપલબ્ધ છે, અંગ્રેજી બતાવવામાં આવે છે.)",
        "pa": "(ਲਾਈਵ ਅਨੁਵਾਦ ਅਸਥਾਈ ਤੌਰ 'ਤੇ ਉਪਲਬਧ ਨਹੀਂ ਹੈ, ਅੰਗਰੇਜ਼ੀ ਦਿਖਾਈ ਜਾ ਰਹੀ ਹੈ।)",
    },
    "error_generic": {
        "en": "Sorry, something went wrong. Please try again.",
        "hi": "क्षमा करें, कुछ गलत हो गया। कृपया पुनः प्रयास करें।",
        "te": "క్షమించండి, ఏదో తప్పు జరిగింది. దయచేసి మళ్లీ ప్రయత్నించండి.",
        "ta": "மன்னிக்கவும், ஏதோ தவறு நடந்தது. மீண்டும் முயற்சிக்கவும்.",
        "kn": "ಕ್ಷಮಿಸಿ, ಏನೋ ತಪ್ಪಾಗಿದೆ. ದಯವಿಟ್ಟು ಮತ್ತೆ ಪ್ರಯತ್ನಿಸಿ.",
        "ml": "ക്ഷമിക്കണം, എന്തോ കുഴപ്പം സംഭവിച്ചു. വീണ്ടും ശ്രമിക്കുക.",
        "bn": "দুঃখিত, কিছু ভুল হয়েছে। আবার চেষ্টা করুন।",
        "mr": "माफ करा, काहीतरी चूक झाली. कृपया पुन्हा प्रयत्न करा.",
        "gu": "માફ કરશો, કંઈક ખોટું થયું. કૃપા કરીને ફરી પ્રયાસ કરો.",
        "pa": "ਮਾਫ਼ ਕਰਨਾ, ਕੁਝ ਗਲਤ ਹੋ ਗਿਆ। ਕਿਰਪਾ ਕਰਕੇ ਦੁਬਾਰਾ ਕੋਸ਼ਿਸ਼ ਕਰੋ।",
    },
    "result_healthy": {
        "en": "Good news - this leaf looks Healthy. No strong Ring Spot Virus pattern was detected.",
        "hi": "अच्छी खबर - यह पत्ता स्वस्थ लग रहा है। रिंग स्पॉट वायरस का कोई मजबूत पैटर्न नहीं मिला।",
        "te": "శుభవార్త - ఈ ఆకు ఆరోగ్యంగా కనిపిస్తోంది. బలమైన రింగ్ స్పాట్ వైరస్ నమూనా కనుగొనబడలేదు.",
        "ta": "நல்ல செய்தி - இந்த இலை ஆரோக்கியமாக தெரிகிறது. வலுவான ரிங் ஸ்பாட் வைரஸ் முறை கண்டறியப்படவில்லை.",
        "kn": "ಒಳ್ಳೆಯ ಸುದ್ದಿ - ಈ ಎಲೆ ಆರೋಗ್ಯಕರವಾಗಿ ಕಾಣುತ್ತದೆ. ಬಲವಾದ ರಿಂಗ್ ಸ್ಪಾಟ್ ವೈರಸ್ ಮಾದರಿ ಪತ್ತೆಯಾಗಿಲ್ಲ.",
        "ml": "സന്തോഷ വാർത്ത - ഈ ഇല ആരോഗ്യമുള്ളതായി കാണപ്പെടുന്നു. ശക്തമായ റിംഗ് സ്പോട്ട് വൈറസ് പാറ്റേൺ കണ്ടെത്തിയില്ല.",
        "bn": "সুখবর - এই পাতাটি সুস্থ দেখাচ্ছে। শক্তিশালী রিং স্পট ভাইরাসের প্যাটার্ন পাওয়া যায়নি।",
        "mr": "चांगली बातमी - हे पान निरोगी दिसत आहे. रिंग स्पॉट व्हायरसचा कोणताही मजबूत नमुना आढळला नाही.",
        "gu": "સારા સમાચાર - આ પાન સ્વસ્થ દેખાય છે. કોઈ મજબૂત રિંગ સ્પોટ વાયરસ પેટર્ન મળી નથી.",
        "pa": "ਚੰਗੀ ਖ਼ਬਰ - ਇਹ ਪੱਤਾ ਸਿਹਤਮੰਦ ਲੱਗ ਰਿਹਾ ਹੈ। ਕੋਈ ਮਜ਼ਬੂਤ ਰਿੰਗ ਸਪਾਟ ਵਾਇਰਸ ਪੈਟਰਨ ਨਹੀਂ ਮਿਲਿਆ।",
    },
    "result_diseased": {
        "en": "This leaf shows a pattern that may be Ring Spot Virus. Please see the details and recommended next steps below.",
        "hi": "इस पत्ते में एक पैटर्न दिखाई देता है जो रिंग स्पॉट वायरस हो सकता है। कृपया नीचे विवरण और अनुशंसित अगले कदम देखें।",
        "te": "ఈ ఆకులో రింగ్ స్పాట్ వైరస్ కావచ్చు అనే నమూనా కనిపిస్తోంది. దయచేసి కింద వివరాలు మరియు సిఫార్సు చేసిన తదుపరి దశలను చూడండి.",
        "ta": "இந்த இலையில் ரிங் ஸ்பாட் வைரஸாக இருக்கக்கூடிய ஒரு முறை காணப்படுகிறது. கீழே உள்ள விவரங்களையும் பரிந்துரைக்கப்பட்ட அடுத்த படிகளையும் பார்க்கவும்.",
        "kn": "ಈ ಎಲೆಯಲ್ಲಿ ರಿಂಗ್ ಸ್ಪಾಟ್ ವೈರಸ್ ಆಗಿರಬಹುದಾದ ಮಾದರಿ ಕಂಡುಬರುತ್ತದೆ. ದಯವಿಟ್ಟು ಕೆಳಗಿನ ವಿವರಗಳು ಮತ್ತು ಶಿಫಾರಸು ಮಾಡಲಾದ ಮುಂದಿನ ಹಂತಗಳನ್ನು ನೋಡಿ.",
        "ml": "ഈ ഇലയിൽ റിംഗ് സ്പോട്ട് വൈറസ് ആകാവുന്ന ഒരു പാറ്റേൺ കാണുന്നു. ദയവായി താഴെയുള്ള വിശദാംശങ്ങളും ശുപാർശ ചെയ്യുന്ന അടുത്ത ഘട്ടങ്ങളും കാണുക.",
        "bn": "এই পাতায় একটি প্যাটার্ন দেখা যাচ্ছে যা রিং স্পট ভাইরাস হতে পারে। অনুগ্রহ করে নীচের বিবরণ এবং প্রস্তাবিত পরবর্তী পদক্ষেপগুলি দেখুন।",
        "mr": "या पानावर एक नमुना दिसतो जो रिंग स्पॉट व्हायरस असू शकतो. कृपया खालील तपशील आणि शिफारस केलेल्या पुढील पायऱ्या पहा.",
        "gu": "આ પાનમાં એક પેટર્ન દેખાય છે જે રિંગ સ્પોટ વાયરસ હોઈ શકે છે. કૃપા કરીને નીચે વિગતો અને ભલામણ કરેલ આગળના પગલાં જુઓ.",
        "pa": "ਇਸ ਪੱਤੇ ਵਿੱਚ ਇੱਕ ਪੈਟਰਨ ਦਿਖਾਈ ਦਿੰਦਾ ਹੈ ਜੋ ਰਿੰਗ ਸਪਾਟ ਵਾਇਰਸ ਹੋ ਸਕਦਾ ਹੈ। ਕਿਰਪਾ ਕਰਕੇ ਹੇਠਾਂ ਵੇਰਵੇ ਅਤੇ ਸਿਫ਼ਾਰਸ਼ ਕੀਤੇ ਅਗਲੇ ਕਦਮ ਵੇਖੋ।",
    },
}


def get_phrase(key: str, lang: str) -> str:
    entry = PHRASEBOOK.get(key, {})
    return entry.get(lang) or entry.get(DEFAULT_LANGUAGE, "")

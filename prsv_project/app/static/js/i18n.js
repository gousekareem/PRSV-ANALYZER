(function () {
    "use strict";

    const LANGUAGES = {
        en: "English",
        hi: "हिन्दी",
        bn: "বাংলা",
        ta: "தமிழ்",
        te: "తెలుగు",
        kn: "ಕನ್ನಡ",
        mr: "मराठी",
        gu: "ગુજરાતી",
        pa: "ਪੰਜਾਬੀ",
        ml: "മലയാളം",
        or: "ଓଡ଼ିଆ"
    };

    // Foundation dictionary for shared navigation and common controls.
    // Additional page dictionaries can be added without changing templates.
    const TEXT = {
        hi: { "Home": "होम", "Analyze": "विश्लेषण", "Runs": "रन इतिहास", "Batch": "बैच", "Status": "स्थिति", "Start Analysis": "विश्लेषण शुरू करें", "View Runs": "रन देखें", "Check System Status": "सिस्टम स्थिति देखें", "No recent runs yet": "अभी कोई हालिया रन नहीं", "No matching runs found": "कोई मिलान रन नहीं मिला", "Start First Analysis": "पहला विश्लेषण शुरू करें", "Start Analysis": "विश्लेषण शुरू करें", "View All Runs": "सभी रन देखें", "Analyze Leaf": "पत्ती का विश्लेषण", "Batch Analysis": "बैच विश्लेषण", "System Status": "सिस्टम स्थिति", "Open": "खोलें", "Apply Filters": "फ़िल्टर लागू करें", "Reset": "रीसेट", "English": "अंग्रेज़ी" },
        bn: { "Home": "হোম", "Analyze": "বিশ্লেষণ", "Runs": "রান ইতিহাস", "Batch": "ব্যাচ", "Status": "স্থিতি", "Start Analysis": "বিশ্লেষণ শুরু করুন", "View Runs": "রান দেখুন", "No recent runs yet": "এখনও কোনো সাম্প্রতিক রান নেই", "Start First Analysis": "প্রথম বিশ্লেষণ শুরু করুন" },
        ta: { "Home": "முகப்பு", "Analyze": "பகுப்பாய்வு", "Runs": "ரன் வரலாறு", "Batch": "தொகுதி", "Status": "நிலை", "Start Analysis": "பகுப்பாய்வைத் தொடங்கவும்", "View Runs": "ரன்களைக் காண்க", "No recent runs yet": "சமீபத்திய ரன்கள் இல்லை", "Start First Analysis": "முதல் பகுப்பாய்வைத் தொடங்கவும்" },
        te: { "Home": "హోమ్", "Analyze": "విశ్లేషణ", "Runs": "రన్ చరిత్ర", "Batch": "బ్యాచ్", "Status": "స్థితి", "Start Analysis": "విశ్లేషణ ప్రారంభించండి", "View Runs": "రన్‌లను చూడండి", "No recent runs yet": "ఇటీవలి రన్‌లు లేవు", "Start First Analysis": "మొదటి విశ్లేషణ ప్రారంభించండి" },
        kn: { "Home": "ಮುಖಪುಟ", "Analyze": "ವಿಶ್ಲೇಷಣೆ", "Runs": "ರನ್ ಇತಿಹಾಸ", "Batch": "ಬ್ಯಾಚ್", "Status": "ಸ್ಥಿತಿ", "Start Analysis": "ವಿಶ್ಲೇಷಣೆ ಪ್ರಾರಂಭಿಸಿ", "View Runs": "ರನ್‌ಗಳನ್ನು ವೀಕ್ಷಿಸಿ" },
        mr: { "Home": "मुख्यपृष्ठ", "Analyze": "विश्लेषण", "Runs": "रन इतिहास", "Batch": "बॅच", "Status": "स्थिती", "Start Analysis": "विश्लेषण सुरू करा", "View Runs": "रन पहा" },
        gu: { "Home": "હોમ", "Analyze": "વિશ્લેષણ", "Runs": "રન ઇતિહાસ", "Batch": "બેચ", "Status": "સ્થિતિ", "Start Analysis": "વિશ્લેષણ શરૂ કરો", "View Runs": "રન જુઓ" },
        pa: { "Home": "ਮੁੱਖ ਪੰਨਾ", "Analyze": "ਵਿਸ਼ਲੇਸ਼ਣ", "Runs": "ਰਨ ਇਤਿਹਾਸ", "Batch": "ਬੈਚ", "Status": "ਸਥਿਤੀ", "Start Analysis": "ਵਿਸ਼ਲੇਸ਼ਣ ਸ਼ੁਰੂ ਕਰੋ", "View Runs": "ਰਨ ਵੇਖੋ" },
        ml: { "Home": "ഹോം", "Analyze": "വിശകലനം", "Runs": "റൺ ചരിത്രം", "Batch": "ബാച്ച്", "Status": "നില", "Start Analysis": "വിശകലനം ആരംഭിക്കുക", "View Runs": "റൺ കാണുക" },
        or: { "Home": "ମୁଖ୍ୟ ପୃଷ୍ଠା", "Analyze": "ବିଶ୍ଳେଷଣ", "Runs": "ରନ୍ ଇତିହାସ", "Batch": "ବ୍ୟାଚ୍", "Status": "ସ୍ଥିତି", "Start Analysis": "ବିଶ୍ଳେଷଣ ଆରମ୍ଭ କରନ୍ତୁ", "View Runs": "ରନ୍ ଦେଖନ୍ତୁ" }
    };

    function detectLanguage() {
        const saved = localStorage.getItem("prsv-language");
        if (saved && LANGUAGES[saved]) return saved;
        const browser = (navigator.language || "en").toLowerCase().split("-")[0];
        return LANGUAGES[browser] ? browser : "en";
    }

    function translateText(lang) {
        document.documentElement.lang = lang;
        document.documentElement.dir = lang === "ur" ? "rtl" : "ltr";
        const dictionary = TEXT[lang] || {};
        document.querySelectorAll("[data-i18n]").forEach((node) => {
            const key = node.getAttribute("data-i18n");
            if (dictionary[key]) node.textContent = dictionary[key];
        });
        document.querySelectorAll("a, button, h1, h2, h3, p, span, label, option").forEach((node) => {
            if (node.children.length || node.closest("script, style, select")) return;
            const key = node.textContent.trim();
            if (dictionary[key]) node.textContent = dictionary[key];
        });
        const selector = document.getElementById("language-selector");
        if (selector) selector.value = lang;
        localStorage.setItem("prsv-language", lang);
    }

    function init() {
        const selector = document.getElementById("language-selector");
        if (selector) {
            Object.entries(LANGUAGES).forEach(([code, name]) => {
                const option = document.createElement("option");
                option.value = code;
                option.textContent = name;
                selector.appendChild(option);
            });
            selector.addEventListener("change", () => translateText(selector.value));
        }
        translateText(detectLanguage());
    }

    window.prsvI18n = { LANGUAGES, translateText, detectLanguage };
    if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
    else init();
})();

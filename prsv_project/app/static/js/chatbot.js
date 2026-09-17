(function () {
    function qs(selector) {
        return document.querySelector(selector);
    }

    const launcher = qs("#chatbot-launcher");
    const panel = qs("#chatbot-panel");
    const closeBtn = qs("#chatbot-close");
    const messagesEl = qs("#chatbot-messages");
    const form = qs("#chatbot-form");
    const textInput = qs("#chatbot-text-input");
    const langSelect = qs("#chatbot-language");
    const micBtn = qs("#chatbot-mic-btn");
    const photoBtn = qs("#chatbot-photo-btn");
    const photoInput = qs("#chatbot-photo-input");

    if (!launcher || !panel) return;

    let greeted = false;
    let recognition = null;
    let isRecording = false;

    // --- Panel open/close ---

    function openPanel() {
        panel.classList.add("open");
        panel.setAttribute("aria-hidden", "false");
        launcher.setAttribute("aria-expanded", "true");
        if (!greeted) {
            greeted = true;
            sendMessage("hello", { silent: false, isGreetingTrigger: true });
        }
        textInput.focus();
    }

    function closePanel() {
        panel.classList.remove("open");
        panel.setAttribute("aria-hidden", "true");
        launcher.setAttribute("aria-expanded", "false");
    }

    launcher.addEventListener("click", () => {
        if (panel.classList.contains("open")) {
            closePanel();
        } else {
            openPanel();
        }
    });
    closeBtn.addEventListener("click", closePanel);

    // --- Message rendering ---

    function appendMessage(role, html) {
        const msg = document.createElement("div");
        msg.className = `chatbot-msg ${role}`;
        msg.innerHTML = html;
        messagesEl.appendChild(msg);
        messagesEl.scrollTop = messagesEl.scrollHeight;
        return msg;
    }

    function appendUserText(text) {
        const div = document.createElement("div");
        div.textContent = text;
        appendMessage("user", div.innerHTML);
    }

    function showTyping() {
        const el = document.createElement("div");
        el.className = "chatbot-typing";
        el.id = "chatbot-typing-indicator";
        el.innerHTML = "<span></span><span></span><span></span>";
        messagesEl.appendChild(el);
        messagesEl.scrollTop = messagesEl.scrollHeight;
    }

    function hideTyping() {
        const el = qs("#chatbot-typing-indicator");
        if (el) el.remove();
    }

    function speak(text, langCode) {
        if (!("speechSynthesis" in window) || !text) return;
        try {
            const utterance = new SpeechSynthesisUtterance(text);
            utterance.lang = langCode || "en-IN";
            window.speechSynthesis.cancel();
            window.speechSynthesis.speak(utterance);
        } catch (e) { /* speech synthesis not usable, ignore */ }
    }

    function currentLanguage() {
        return langSelect ? langSelect.value : "en";
    }

    // --- Sending text messages ---

    function sendMessage(text, opts) {
        opts = opts || {};
        if (!opts.isGreetingTrigger) {
            appendUserText(text);
        }
        showTyping();

        fetch("/api/chatbot/message", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text: text, language: currentLanguage() }),
        })
            .then((res) => res.json())
            .then((data) => {
                hideTyping();
                appendMessage("bot", escapeHtml(data.reply));
                speak(data.reply, speechLocaleFor(data.language));
            })
            .catch(() => {
                hideTyping();
                appendMessage("bot", "Sorry, I couldn't reach the server. Please try again.");
            });
    }

    const SPEECH_LOCALES = {
        en: "en-IN", hi: "hi-IN", te: "te-IN", ta: "ta-IN", kn: "kn-IN",
        ml: "ml-IN", bn: "bn-IN", mr: "mr-IN", gu: "gu-IN", pa: "pa-IN",
    };

    function speechLocaleFor(langCode) {
        return SPEECH_LOCALES[langCode] || "en-IN";
    }

    function escapeHtml(str) {
        const div = document.createElement("div");
        div.textContent = str;
        return div.innerHTML;
    }

    form.addEventListener("submit", (e) => {
        e.preventDefault();
        const text = textInput.value.trim();
        if (!text) return;
        textInput.value = "";
        sendMessage(text);
    });

    // --- Voice input (Web Speech API - graceful fallback if unsupported) ---

    const SpeechRecognitionCtor = window.SpeechRecognition || window.webkitSpeechRecognition;

    if (!SpeechRecognitionCtor) {
        micBtn.classList.add("unsupported");
        micBtn.title = "Voice input isn't supported in this browser";
    } else {
        micBtn.addEventListener("click", () => {
            if (isRecording && recognition) {
                recognition.stop();
                return;
            }

            recognition = new SpeechRecognitionCtor();
            recognition.lang = speechLocaleFor(currentLanguage());
            recognition.interimResults = false;
            recognition.maxAlternatives = 1;

            recognition.onstart = () => {
                isRecording = true;
                micBtn.classList.add("recording");
            };

            recognition.onresult = (event) => {
                const transcript = event.results[0][0].transcript;
                textInput.value = transcript;
                sendMessage(transcript);
            };

            recognition.onerror = () => {
                window.PRSVApp && window.PRSVApp.showToast("Couldn't hear that clearly. Please try again or type instead.", "warning");
            };

            recognition.onend = () => {
                isRecording = false;
                micBtn.classList.remove("recording");
            };

            try {
                recognition.start();
            } catch (e) {
                isRecording = false;
                micBtn.classList.remove("recording");
            }
        });
    }

    // --- Photo-in-chat diagnosis ---

    photoBtn.addEventListener("click", () => photoInput.click());

    photoInput.addEventListener("change", () => {
        const file = photoInput.files && photoInput.files[0];
        if (!file) return;

        const previewUrl = URL.createObjectURL(file);
        appendMessage("user", `<img src="${previewUrl}" alt="Uploaded leaf photo">`);
        showTyping();

        const formData = new FormData();
        formData.append("file", file);

        fetch(`/api/chatbot/analyze-image?language=${encodeURIComponent(currentLanguage())}`, {
            method: "POST",
            body: formData,
        })
            .then((res) => res.json())
            .then((data) => {
                hideTyping();
                if (data.error) {
                    appendMessage("bot", escapeHtml(data.reply || "Sorry, I couldn't process that photo."));
                    return;
                }

                const msg = appendMessage("bot", escapeHtml(data.reply));
                if (data.run_id && data.image_id) {
                    const actions = document.createElement("div");
                    actions.className = "chatbot-msg-actions";
                    const link = document.createElement("a");
                    link.href = `/run/${data.run_id}/image/${data.image_id}`;
                    link.target = "_blank";
                    link.rel = "noopener";
                    link.textContent = "View full details";
                    actions.appendChild(link);
                    msg.appendChild(actions);
                }
                speak(data.reply, speechLocaleFor(data.language));
            })
            .catch(() => {
                hideTyping();
                appendMessage("bot", "Sorry, something went wrong while checking that photo. Please try again.");
            })
            .finally(() => {
                photoInput.value = "";
            });
    });
})();

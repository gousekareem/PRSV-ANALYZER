(function () {
    function qs(selector) {
        return document.querySelector(selector);
    }

    function qsa(selector) {
        return Array.from(document.querySelectorAll(selector));
    }

    const toastRoot = qs("#toast-root");
    const loadingOverlay = qs("#global-loading-overlay");
    const loadingTitle = qs("#loading-title");
    const loadingMessage = qs("#loading-message");
    const loadingProgressBar = qs("#loading-progress-bar");
    const loadingSteps = qs("#loading-steps");

    let loadingInterval = null;

    function hideLoadingOverlay() {
        if (loadingInterval) {
            window.clearInterval(loadingInterval);
            loadingInterval = null;
        }

        if (loadingOverlay) {
            loadingOverlay.style.display = "none";
        }

        document.body.classList.remove("overlay-open");

        const activeButtons = qsa("button.is-loading");
        activeButtons.forEach((button) => {
            button.disabled = false;
            if (button.dataset.originalText) {
                button.textContent = button.dataset.originalText;
            }
            button.classList.remove("is-loading");
        });
    }

    function showToast(message, type = "info", timeout = 3500) {
        if (!toastRoot) return;

        const toast = document.createElement("div");
        toast.className = `toast toast-${type}`;
        toast.innerHTML = `
            <div class="toast-title">${type.charAt(0).toUpperCase() + type.slice(1)}</div>
            <div class="toast-text">${message}</div>
        `;

        toastRoot.appendChild(toast);

        requestAnimationFrame(() => {
            toast.classList.add("show");
        });

        window.setTimeout(() => {
            toast.classList.remove("show");
            window.setTimeout(() => {
                toast.remove();
            }, 220);
        }, timeout);
    }

    function setLoadingStep(index) {
        if (!loadingSteps) return;
        const items = qsa("#loading-steps li");

        items.forEach((item, idx) => {
            item.classList.remove("active", "done");
            if (idx < index) item.classList.add("done");
            if (idx === index) item.classList.add("active");
        });

        if (loadingProgressBar) {
            const progress = Math.min(100, Math.round(((index + 1) / Math.max(items.length, 1)) * 100));
            loadingProgressBar.style.width = `${progress}%`;
        }
    }

    function showLoadingSequence(config) {
        if (!loadingOverlay) return;

        const steps = config.steps || [
            "Uploading files",
            "Validating inputs",
            "Running analysis",
            "Preparing results",
        ];

        if (loadingTitle) loadingTitle.textContent = config.title || "Processing request";
        if (loadingMessage) loadingMessage.textContent = config.message || "Please wait while PRSV ANALYZER processes your request.";

        if (loadingSteps) {
            loadingSteps.innerHTML = "";
            steps.forEach((step, idx) => {
                const li = document.createElement("li");
                li.textContent = step;
                if (idx === 0) li.classList.add("active");
                loadingSteps.appendChild(li);
            });
        }

        if (loadingProgressBar) {
            loadingProgressBar.style.width = "12%";
        }

        loadingOverlay.style.display = "grid";
        document.body.classList.add("overlay-open");

        let currentStep = 0;
        loadingInterval = window.setInterval(() => {
            currentStep += 1;
            if (currentStep >= steps.length) {
                currentStep = steps.length - 1;
            }
            setLoadingStep(currentStep);
        }, config.stepDuration || 900);
    }

    function bindAnalysisForms() {
        const forms = qsa("form[data-analysis-form='true']");
        if (!forms.length) return;

        forms.forEach((form) => {
            form.addEventListener("submit", () => {
                const button = form.querySelector("button[type='submit']");
                if (button) {
                    button.disabled = true;
                    button.dataset.originalText = button.textContent;
                    button.textContent = "Working...";
                    button.classList.add("is-loading");
                }

                let steps = [
                    "Uploading files",
                    "Validating inputs",
                    "Running analysis",
                    "Preparing results",
                ];

                const mode = form.dataset.analysisMode || "single";

                if (mode === "single") {
                    steps = [
                        "Uploading image",
                        "Validating file",
                        "Preprocessing image",
                        "Running disease analysis",
                        "Preparing result view",
                    ];
                } else if (mode === "batch") {
                    steps = [
                        "Uploading images",
                        "Validating files",
                        "Processing batch",
                        "Generating charts",
                        "Preparing dashboard",
                    ];
                } else if (mode === "zip") {
                    steps = [
                        "Uploading archive",
                        "Validating ZIP contents",
                        "Extracting images",
                        "Processing batch",
                        "Preparing dashboard",
                    ];
                } else if (mode === "demo") {
                    steps = [
                        "Loading demo dataset",
                        "Preparing analysis queue",
                        "Processing demo images",
                        "Generating run outputs",
                        "Opening dashboard",
                    ];
                }

                showLoadingSequence({
                    title: "Analysis in progress",
                    message: "PRSV ANALYZER is preparing your results.",
                    steps,
                    stepDuration: 850,
                });
            });
        });
    }

    function bindCopyActions() {
        const buttons = qsa("[data-copy-text]");
        buttons.forEach((button) => {
            button.addEventListener("click", async () => {
                const text = button.getAttribute("data-copy-text") || "";
                try {
                    await navigator.clipboard.writeText(text);
                    showToast("Copied successfully", "success", 2200);
                } catch (error) {
                    showToast("Copy failed", "error", 2200);
                }
            });
        });
    }

    function boot() {
        hideLoadingOverlay();
        bindAnalysisForms();
        bindCopyActions();
    }

    document.addEventListener("DOMContentLoaded", boot);
    window.addEventListener("pageshow", hideLoadingOverlay);
    window.addEventListener("load", hideLoadingOverlay);

    window.PRSVApp = {
        showToast,
        showLoadingSequence,
        hideLoadingOverlay,
    };
})();
(function () {
    function qs(selector, root) {
        return (root || document).querySelector(selector);
    }
    function qsa(selector, root) {
        return Array.from((root || document).querySelectorAll(selector));
    }

    function readJsonScript(id) {
        const el = document.getElementById(id);
        if (!el) return null;
        try {
            return JSON.parse(el.textContent);
        } catch (e) {
            return null;
        }
    }

    function isDarkMode() {
        return document.documentElement.getAttribute("data-theme") === "dark";
    }

    function chartTextColor() {
        return isDarkMode() ? "#e9f0e7" : "#18221b";
    }

    // --- Run dashboard: prediction + severity charts ---

    function renderRunCharts() {
        const data = readJsonScript("run-chart-data");
        if (!data || typeof window.Chart === "undefined") return;

        const predictionCanvas = qs("#prediction-chart");
        if (predictionCanvas) {
            new window.Chart(predictionCanvas, {
                type: "doughnut",
                data: {
                    labels: ["Healthy", "PRSV Suspected"],
                    datasets: [{
                        data: [data.healthy_count || 0, data.diseased_count || 0],
                        backgroundColor: ["#2f855a", "#c53030"],
                        borderWidth: 0,
                    }],
                },
                options: {
                    plugins: {
                        legend: { labels: { color: chartTextColor() } },
                    },
                },
            });
        }

        const severityCanvas = qs("#severity-chart");
        if (severityCanvas && data.severity_distribution) {
            const labels = Object.keys(data.severity_distribution);
            const values = Object.values(data.severity_distribution);
            new window.Chart(severityCanvas, {
                type: "bar",
                data: {
                    labels,
                    datasets: [{
                        label: "Images",
                        data: values,
                        backgroundColor: "#2f855a",
                        borderRadius: 6,
                    }],
                },
                options: {
                    plugins: { legend: { display: false } },
                    scales: {
                        x: { ticks: { color: chartTextColor() } },
                        y: { ticks: { color: chartTextColor() }, beginAtZero: true, precision: 0 },
                    },
                },
            });
        }
    }

    // --- Image detail: SHAP contribution bars ---

    function renderShapBars() {
        const data = readJsonScript("shap-data");
        const container = qs("#shap-bars");
        if (!data || !container) return;

        const entries = Object.entries(data).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));
        const maxAbs = Math.max(...entries.map(([, v]) => Math.abs(v)), 0.001);

        container.innerHTML = "";
        entries.forEach(([name, value]) => {
            const row = document.createElement("div");
            row.className = "shap-bar-row";

            const label = document.createElement("div");
            label.className = "shap-bar-label";
            label.textContent = name.replace(/_/g, " ");
            row.appendChild(label);

            const track = document.createElement("div");
            track.className = "shap-bar-track";

            const mid = document.createElement("div");
            mid.className = "shap-bar-mid";
            track.appendChild(mid);

            const fill = document.createElement("div");
            fill.className = `shap-bar-fill ${value >= 0 ? "positive" : "negative"}`;
            const widthPct = (Math.abs(value) / maxAbs) * 48;
            fill.style.width = `${widthPct}%`;
            track.appendChild(fill);

            row.appendChild(track);
            container.appendChild(row);
        });
    }

    // --- Before/after comparison slider ---

    function setupCompareSliders() {
        qsa(".compare-slider").forEach((slider) => {
            const handle = slider.querySelector(".compare-handle");
            const after = slider.querySelector(".compare-after");
            if (!handle || !after) return;

            let dragging = false;

            function setPosition(clientX) {
                const rect = slider.getBoundingClientRect();
                let pct = ((clientX - rect.left) / rect.width) * 100;
                pct = Math.max(0, Math.min(100, pct));
                after.style.clipPath = `inset(0 0 0 ${pct}%)`;
                handle.style.left = `${pct}%`;
            }

            handle.addEventListener("mousedown", () => { dragging = true; });
            window.addEventListener("mouseup", () => { dragging = false; });
            window.addEventListener("mousemove", (e) => {
                if (dragging) setPosition(e.clientX);
            });

            handle.addEventListener("touchstart", () => { dragging = true; }, { passive: true });
            window.addEventListener("touchend", () => { dragging = false; });
            window.addEventListener("touchmove", (e) => {
                if (dragging && e.touches[0]) setPosition(e.touches[0].clientX);
            }, { passive: true });

            slider.addEventListener("click", (e) => setPosition(e.clientX));
        });
    }

    // --- Lightbox for result images ---

    function setupLightbox() {
        const images = qsa(".lightbox-trigger");
        if (!images.length) return;

        const overlay = document.createElement("div");
        overlay.className = "lightbox-overlay";
        overlay.innerHTML = `
            <button class="lightbox-close" type="button" aria-label="Close image preview">✕</button>
            <img alt="">
        `;
        document.body.appendChild(overlay);

        const overlayImg = overlay.querySelector("img");
        const closeBtn = overlay.querySelector(".lightbox-close");

        function open(src, alt) {
            overlayImg.src = src;
            overlayImg.alt = alt || "";
            overlay.classList.add("open");
        }
        function close() {
            overlay.classList.remove("open");
        }

        images.forEach((img) => {
            img.addEventListener("click", () => open(img.src, img.alt));
        });
        closeBtn.addEventListener("click", close);
        overlay.addEventListener("click", (e) => {
            if (e.target === overlay) close();
        });
        document.addEventListener("keydown", (e) => {
            if (e.key === "Escape") close();
        });
    }

    // --- Client-side pagination for run grids (Runs / Batch pages) ---

    function setupGridPagination() {
        const grid = qs("[data-paginated-grid]");
        if (!grid) return;

        const pageSize = parseInt(grid.getAttribute("data-page-size") || "9", 10);
        const cards = qsa(":scope > *", grid);
        if (cards.length <= pageSize) return;

        const paginationEl = document.createElement("div");
        paginationEl.className = "pagination";
        grid.insertAdjacentElement("afterend", paginationEl);

        const totalPages = Math.ceil(cards.length / pageSize);
        let currentPage = 1;

        function render() {
            cards.forEach((card, idx) => {
                const page = Math.floor(idx / pageSize) + 1;
                card.style.display = page === currentPage ? "" : "none";
            });

            paginationEl.innerHTML = "";

            const prevBtn = document.createElement("button");
            prevBtn.textContent = "‹";
            prevBtn.disabled = currentPage === 1;
            prevBtn.addEventListener("click", () => { currentPage -= 1; render(); });
            paginationEl.appendChild(prevBtn);

            for (let i = 1; i <= totalPages; i += 1) {
                const btn = document.createElement("button");
                btn.textContent = String(i);
                if (i === currentPage) btn.classList.add("active");
                btn.addEventListener("click", () => { currentPage = i; render(); });
                paginationEl.appendChild(btn);
            }

            const nextBtn = document.createElement("button");
            nextBtn.textContent = "›";
            nextBtn.disabled = currentPage === totalPages;
            nextBtn.addEventListener("click", () => { currentPage += 1; render(); });
            paginationEl.appendChild(nextBtn);
        }

        render();
    }

    // --- Live client-side search-as-you-type on run grids ---

    function setupLiveSearch() {
        const input = qs("[data-live-search]");
        const grid = qs("[data-paginated-grid]");
        if (!input || !grid) return;

        input.addEventListener("input", () => {
            const term = input.value.trim().toLowerCase();
            qsa(":scope > *", grid).forEach((card) => {
                const text = card.textContent.toLowerCase();
                card.style.display = !term || text.includes(term) ? "" : "none";
            });
        });
    }

    document.addEventListener("DOMContentLoaded", () => {
        renderRunCharts();
        renderShapBars();
        setupCompareSliders();
        setupLightbox();
        setupGridPagination();
        setupLiveSearch();
    });
})();

(function () {
    function qs(selector, root) {
        return (root || document).querySelector(selector);
    }

    const ALLOWED_TYPES = ["image/jpeg", "image/png", "image/jpg", "image/bmp", "image/tiff", "image/webp"];
    const MAX_FILE_MB = 25;

    function isValidImageFile(file) {
        if (file.size === 0) return { ok: false, reason: `${file.name}: file is empty.` };
        if (file.size > MAX_FILE_MB * 1024 * 1024) {
            return { ok: false, reason: `${file.name}: exceeds ${MAX_FILE_MB}MB limit.` };
        }
        if (ALLOWED_TYPES.length && file.type && !ALLOWED_TYPES.includes(file.type)) {
            return { ok: false, reason: `${file.name}: unsupported file type.` };
        }
        return { ok: true };
    }

    function formatBytes(bytes) {
        if (bytes < 1024) return `${bytes} B`;
        if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
        return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
    }

    function setupDropzone(config) {
        const dropzone = qs(config.dropzoneSelector);
        const input = qs(config.inputSelector);
        const previewGrid = qs(config.previewSelector);
        if (!dropzone || !input) return null;

        let files = [];

        function render() {
            if (!previewGrid) return;
            previewGrid.innerHTML = "";
            files.forEach((file, idx) => {
                const tile = document.createElement("div");
                tile.className = "preview-tile";

                const img = document.createElement("img");
                img.src = URL.createObjectURL(file);
                img.alt = file.name;
                img.onload = () => URL.revokeObjectURL(img.src);
                tile.appendChild(img);

                const nameTag = document.createElement("div");
                nameTag.className = "preview-filename";
                nameTag.textContent = `${file.name} (${formatBytes(file.size)})`;
                tile.appendChild(nameTag);

                if (config.multiple) {
                    const removeBtn = document.createElement("button");
                    removeBtn.type = "button";
                    removeBtn.className = "preview-remove";
                    removeBtn.setAttribute("aria-label", `Remove ${file.name}`);
                    removeBtn.textContent = "✕";
                    removeBtn.addEventListener("click", () => {
                        files.splice(idx, 1);
                        render();
                    });
                    tile.appendChild(removeBtn);
                }

                previewGrid.appendChild(tile);
            });
        }

        function addFiles(fileList) {
            const incoming = Array.from(fileList);
            const errors = [];

            incoming.forEach((file) => {
                const check = isValidImageFile(file);
                if (!check.ok) {
                    errors.push(check.reason);
                    return;
                }
                if (!config.multiple) {
                    files = [file];
                } else {
                    files.push(file);
                }
            });

            if (errors.length && window.PRSVApp) {
                window.PRSVApp.showToast(errors.join(" "), "error", 4500);
            }

            render();
        }

        dropzone.addEventListener("click", () => input.click());
        dropzone.addEventListener("keydown", (e) => {
            if (e.key === "Enter" || e.key === " ") {
                e.preventDefault();
                input.click();
            }
        });

        input.addEventListener("change", () => {
            if (input.files && input.files.length) addFiles(input.files);
        });

        ["dragenter", "dragover"].forEach((evt) => {
            dropzone.addEventListener(evt, (e) => {
                e.preventDefault();
                dropzone.classList.add("dragover");
            });
        });

        ["dragleave", "drop"].forEach((evt) => {
            dropzone.addEventListener(evt, (e) => {
                e.preventDefault();
                dropzone.classList.remove("dragover");
            });
        });

        dropzone.addEventListener("drop", (e) => {
            if (e.dataTransfer && e.dataTransfer.files.length) addFiles(e.dataTransfer.files);
        });

        return {
            getFiles: () => files,
            clear: () => {
                files = [];
                render();
            },
        };
    }

    function submitWithProgress(url, formData, progressEls, onDone) {
        const xhr = new XMLHttpRequest();
        xhr.open("POST", url, true);

        xhr.upload.onprogress = (e) => {
            if (!e.lengthComputable) return;
            const pct = Math.round((e.loaded / e.total) * 100);
            updateProgress(progressEls, pct, `Uploading... ${pct}%`);
        };

        xhr.onload = () => {
            if (xhr.status >= 200 && xhr.status < 400) {
                onDone(null, xhr);
            } else {
                let detail = "Upload failed.";
                try {
                    detail = JSON.parse(xhr.responseText).detail || detail;
                } catch (e) { /* not JSON */ }
                onDone(detail, xhr);
            }
        };

        xhr.onerror = () => onDone("Network error during upload.", xhr);

        xhr.send(formData);
        return xhr;
    }

    function updateProgress(els, pct, label) {
        if (!els) return;
        if (els.wrap) els.wrap.classList.add("active");
        if (els.fill) els.fill.style.width = `${pct}%`;
        if (els.label) els.label.textContent = label;
    }

    function pollJob(jobId, progressEls, onDone) {
        const poll = () => {
            fetch(`/api/analysis/job/${jobId}`)
                .then((res) => res.json())
                .then((job) => {
                    const total = job.total_images || 1;
                    const done = job.processed_images || 0;
                    const pct = Math.min(100, Math.round((done / total) * 100));

                    if (job.status === "queued") {
                        updateProgress(progressEls, 5, "Queued, starting shortly...");
                        window.setTimeout(poll, 600);
                    } else if (job.status === "processing") {
                        updateProgress(progressEls, Math.max(pct, 10), `Processing ${done} of ${total} image(s)...`);
                        window.setTimeout(poll, 600);
                    } else if (job.status === "done") {
                        updateProgress(progressEls, 100, "Done - opening results...");
                        onDone(null, job);
                    } else {
                        onDone(job.error || "Processing failed.", job);
                    }
                })
                .catch(() => {
                    onDone("Lost connection while checking job progress.", null);
                });
        };
        poll();
    }

    function getProgressEls(form) {
        return {
            wrap: form.querySelector(".upload-progress-wrap"),
            fill: form.querySelector(".upload-progress-fill"),
            label: form.querySelector(".upload-progress-label"),
        };
    }

    function bindSingleForm() {
        const form = qs("#single-upload-form");
        if (!form) return;

        const dz = setupDropzone({
            dropzoneSelector: "#single-dropzone",
            inputSelector: "#single-file-input",
            previewSelector: "#single-preview-grid",
            multiple: false,
        });

        form.addEventListener("submit", (e) => {
            e.preventDefault();
            const files = dz ? dz.getFiles() : [];
            if (!files.length) {
                window.PRSVApp.showToast("Please choose or drop a leaf photo first.", "error");
                return;
            }

            const formData = new FormData();
            formData.append("file", files[0]);

            const progressEls = getProgressEls(form);
            const submitBtn = form.querySelector("button[type='submit']");
            if (submitBtn) submitBtn.disabled = true;

            submitWithProgress(form.action, formData, progressEls, (err, xhr) => {
                if (err) {
                    if (submitBtn) submitBtn.disabled = false;
                    window.PRSVApp.showToast(err, "error");
                    return;
                }
                window.location.href = xhr.responseURL || "/analyze";
            });
        });
    }

    function bindMultiForm() {
        const form = qs("#multi-upload-form");
        if (!form) return;

        const dz = setupDropzone({
            dropzoneSelector: "#multi-dropzone",
            inputSelector: "#multi-file-input",
            previewSelector: "#multi-preview-grid",
            multiple: true,
        });

        form.addEventListener("submit", (e) => {
            e.preventDefault();
            const files = dz ? dz.getFiles() : [];
            if (!files.length) {
                window.PRSVApp.showToast("Please choose or drop at least one image.", "error");
                return;
            }

            const formData = new FormData();
            files.forEach((f) => formData.append("files", f));

            const progressEls = getProgressEls(form);
            const submitBtn = form.querySelector("button[type='submit']");
            if (submitBtn) submitBtn.disabled = true;

            submitWithProgress("/api/analysis/multiple-async", formData, progressEls, (err, xhr) => {
                if (err) {
                    if (submitBtn) submitBtn.disabled = false;
                    window.PRSVApp.showToast(err, "error");
                    return;
                }
                const data = JSON.parse(xhr.responseText);
                updateProgress(progressEls, 0, "Processing batch...");
                pollJob(data.job_id, progressEls, (pollErr, job) => {
                    if (pollErr) {
                        if (submitBtn) submitBtn.disabled = false;
                        window.PRSVApp.showToast(pollErr, "error");
                        return;
                    }
                    window.location.href = `/run/${job.run_id}`;
                });
            });
        });
    }

    function bindZipForm() {
        const form = qs("#zip-upload-form");
        if (!form) return;

        form.addEventListener("submit", (e) => {
            e.preventDefault();
            const input = qs("#zip-file-input");
            if (!input || !input.files.length) {
                window.PRSVApp.showToast("Please choose a ZIP file first.", "error");
                return;
            }

            const formData = new FormData();
            formData.append("file", input.files[0]);

            const progressEls = getProgressEls(form);
            const submitBtn = form.querySelector("button[type='submit']");
            if (submitBtn) submitBtn.disabled = true;

            submitWithProgress("/api/analysis/zip-async", formData, progressEls, (err, xhr) => {
                if (err) {
                    if (submitBtn) submitBtn.disabled = false;
                    window.PRSVApp.showToast(err, "error");
                    return;
                }
                const data = JSON.parse(xhr.responseText);
                updateProgress(progressEls, 0, "Extracting and processing archive...");
                pollJob(data.job_id, progressEls, (pollErr, job) => {
                    if (pollErr) {
                        if (submitBtn) submitBtn.disabled = false;
                        window.PRSVApp.showToast(pollErr, "error");
                        return;
                    }
                    window.location.href = `/run/${job.run_id}`;
                });
            });
        });
    }

    function bindDemoForm() {
        const form = qs("#demo-analysis-form");
        if (!form) return;

        form.addEventListener("submit", (e) => {
            e.preventDefault();
            const limitInput = form.querySelector("input[name='limit']");
            const limit = limitInput ? limitInput.value : 10;

            const progressEls = getProgressEls(form);
            const submitBtn = form.querySelector("button[type='submit']");
            if (submitBtn) submitBtn.disabled = true;
            updateProgress(progressEls, 5, "Starting demo run...");

            fetch(`/api/analysis/demo-async?limit=${encodeURIComponent(limit)}`, { method: "POST" })
                .then((res) => res.json())
                .then((data) => {
                    pollJob(data.job_id, progressEls, (pollErr, job) => {
                        if (pollErr) {
                            if (submitBtn) submitBtn.disabled = false;
                            window.PRSVApp.showToast(pollErr, "error");
                            return;
                        }
                        window.location.href = `/run/${job.run_id}`;
                    });
                })
                .catch(() => {
                    if (submitBtn) submitBtn.disabled = false;
                    window.PRSVApp.showToast("Could not start demo run.", "error");
                });
        });
    }

    document.addEventListener("DOMContentLoaded", () => {
        bindSingleForm();
        bindMultiForm();
        bindZipForm();
        bindDemoForm();
    });
})();

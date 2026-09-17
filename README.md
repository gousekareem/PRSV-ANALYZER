# PRSV Analyzer v3.1 – Papaya Ring Spot Virus Diagnostic System

<div align="center">

![Version](https://img.shields.io/badge/Version-3.1.0-purple.svg)
![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-teal.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-SVM-orange.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-Image%20Processing-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**A research/demo web app that detects Papaya Ring Spot Virus (PRSV) from leaf images: classic image processing + a handcrafted-feature SVM with texture/SHAP explainability + a multilingual farmer chatbot, all in one web app (desktop and mobile browser).**

</div>

---

## Table of Contents

- [Overview](#overview)
- [What This Actually Is (read this first)](#what-this-actually-is-read-this-first)
- [System Architecture](#system-architecture)
- [Technical Pipeline](#technical-pipeline)
- [Project Structure](#project-structure)
- [Installation & Running It (Windows / VS Code)](#installation--running-it-windows--vs-code)
- [Training / Retraining the Model](#training--retraining-the-model)
- [Deploying to Render (Free)](#deploying-to-render-free)
- [Farmer Chatbot](#farmer-chatbot)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [Known Limitations](#known-limitations)
- [Security Notes](#security-notes)
- [Future Scope](#future-scope)
- [Changelog](#changelog)
- [License](#license)

---

## Overview

PRSV Analyzer takes a papaya leaf photo and produces:

- **Prediction** — Healthy / Diseased
- **Severity estimate** — 0–100%, bucketed into a label (Very mild → Severe), suppressed to "Healthy" whenever the prediction is Healthy so the UI never shows a contradictory state like "Healthy – Mild severity"
- **Heatmap overlays** — edge/gradient/symptom visualization
- **A local, template-based explanation** — retrieved from a small curated knowledge base, not an LLM call

This is a research/demo system, not a validated diagnostic tool. See [Known Limitations](#known-limitations) before using it for anything beyond experimentation.

---

## What This Actually Is (read this first)

An earlier version of this README described a Flask app with `app.py`, `models/svm_model.pkl`, and a 5-feature model. **That was never accurate for this codebase.** The real system is:

| README used to say | What's actually here |
|---|---|
| Flask, `app.py` | **FastAPI**, `app/main.py` |
| `templates/index.html`, `utils/` | `app/templates/`, `app/services/`, `image_processing/` |
| `models/svm_model.pkl` | `models/svm_model.joblib` (+ `scaler.joblib`, `label_encoder.joblib`) |
| 5 features | **7 handcrafted features**: `brightness`, `green_ratio`, `hue_mean`, `saturation_mean`, `edge_density`, `color_variance`, `entropy` |
| "RAG" implies an LLM | TF-IDF retrieval (`rag/retriever.py`) + f-string templates (`rag/generator.py`). No LLM call. |

This zip ships with a **real trained model** in `prsv_project/models/` (see below) — it is no longer running on the heuristic fallback by default. If you ever see `"inference_mode": "heuristic_fallback"` in a response, or the startup banner about a missing model, it means `models/svm_model.joblib` couldn't be loaded and you should retrain (see [Training](#training--retraining-the-model)).

---

## System Architecture

```
Upload / Zip / Demo dataset
        |
Preprocess (resize, CLAHE, denoise)
        |
Leaf segmentation (HSV threshold + morphology + largest contour)
        |
Symptom enhancement (Canny / Sobel / Laplacian)
        |
Feature extraction -> 7-value vector
        |            \
        |             -> Heatmap generation (edge/gradient/symptom overlay)
        v
SVM inference (trained_model, or heuristic_fallback if no model files)
        |
Severity estimate (suppressed to 0 / "Healthy" if prediction == Healthy)
        |
Local RAG: TF-IDF retrieval over rag/kb/prsv_knowledge.json -> templated explanation
        |
Saved to data/outputs/run_<timestamp>/images/<image_id>/ (JSON + images)
```

---

## Technical Pipeline

1. **Preprocessing** (`image_processing/preprocess.py`) — resize, optional CLAHE contrast enhancement, optional denoising.
2. **Segmentation** (`image_processing/segmentation.py`) — HSV color thresholding + morphological open/close + largest-contour extraction. Falls back gracefully if no leaf-shaped contour is found.
3. **Symptom enhancement** (`image_processing/symptom_enhancement.py`) — Canny edges, gradient magnitude, Laplacian response, combined into a symptom mask.
4. **Feature extraction** (`image_processing/feature_extraction.py`) — 7 global features (see table below).
5. **Classification** (`ml/infer_svm.py`) — trained RBF-kernel SVM (`sklearn.svm.SVC`) if `models/svm_model.joblib` exists; otherwise a hand-tuned weighted-formula fallback, clearly labeled `inference_mode: "heuristic_fallback"`.
6. **Severity** (`image_processing/severity.py`) — weighted combination of inverse green ratio, edge density, entropy, abnormal color score, and symptom-region ratio → 0–100%, forced to the Healthy band whenever the classifier says Healthy.
7. **Heatmaps** (`image_processing/heatmaps.py`) — edge and severity overlays for visual inspection.
8. **Explanation** (`rag/`) — TF-IDF retrieval over a small curated PRSV knowledge base, spliced into f-string templates. Genuinely useful for surfacing relevant passages, but it is retrieval + templating, not generation.

### Feature table

| Feature | What it measures |
|---|---|
| `brightness` | Mean pixel intensity |
| `green_ratio` | Proportion of leaf pixels in the healthy-green HSV range |
| `hue_mean` | Average hue across the segmented leaf |
| `saturation_mean` | Average color saturation |
| `edge_density` | Density of Canny-detected edges (lesion boundaries) |
| `color_variance` | Spread of color values across the leaf |
| `entropy` | Shannon entropy of the grayscale image (texture irregularity) |

These are global, per-image averages — they don't localize individual ring-spot lesions. See [Known Limitations](#known-limitations).

---

## Project Structure

```
PRSV-ANALYZER/
├── run.bat                        # one-click Windows launcher
├── README.md                      # this file
├── Original Images/                # demo dataset (761 images: Healthy(n).jpg / RingSpot(n).jpg)
└── prsv_project/
    ├── app/
    │   ├── main.py                 # FastAPI app entry point (lifespan, scheduled cleanup, SHAP warm-up)
    │   ├── config.py               # Settings (env-driven, portable defaults)
    │   ├── middleware.py            # NEW (v2.9): optional API key gate + rate limiter
    │   ├── routes/                 # api_analysis, api_chatbot (NEW), api_health, pages
    │   ├── services/                # analysis/batch/dataset/export/rag/run_manager
    │   │   ├── run_store.py                # NEW (v2.9): SQLite run index (search/pagination)
    │   │   ├── job_queue.py                # NEW (v2.9): in-process async batch job queue
    │   │   ├── report_service.py           # NEW (v2.9): PDF report generation (reportlab)
    │   │   ├── cleanup_service.py          # NEW (v2.9): shared cleanup logic (manual + scheduled)
    │   │   ├── chatbot_service.py          # NEW (v2.9): chatbot brain (KB retrieval + photo diagnosis)
    │   │   └── translation_service.py      # NEW (v2.9): languages, translation, offline phrasebook
    │   ├── templates/               # Jinja2 server-rendered pages
    │   │   └── partials/chatbot_widget.html  # NEW (v2.9): chat widget markup, included on every page
    │   ├── static/
    │   │   ├── css/chatbot.css              # NEW (v2.9)
    │   │   └── js/{upload,dashboard,chatbot}.js  # NEW (v2.9): were empty 0-byte files before
    │   └── utils/                   # file/zip/path/validation helpers
    ├── image_processing/            # preprocess, segmentation, features, severity, heatmaps
    │   ├── texture_features.py            # GLCM + LBP texture features
    │   ├── severity_v2.py                 # NEW (v3.0): lesion-area-ratio severity + bootstrap interval
    │   ├── texture_extended.py            # NEW (v3.1): multi-radius LBP, Gabor, Tamura, wavelet, LPQ
    │   ├── color_spaces_extended.py       # NEW (v3.1): Lab/YCbCr/HSI, vegetation indices, color moments
    │   ├── illumination_normalization.py  # NEW (v3.1): Retinex, shadow detection, white balance
    │   └── segmentation_classical.py      # NEW (v3.1): GrabCut, Watershed, SLIC alternatives
    ├── ml/
    │   ├── train_svm.py             # GridSearchCV + StratifiedKFold training (now 12 features)
    │   ├── train_ensemble.py        # NEW (v3.0): stacked SVM+RF+GB ensemble + McNemar comparison
    │   ├── infer_svm.py             # trained-model / heuristic-fallback inference
    │   ├── shap_explainer.py               # per-feature SHAP contributions
    │   ├── calibration.py                  # NEW (v3.0): CalibratedClassifierCV (Platt/isotonic)
    │   ├── ensemble.py                     # NEW (v3.0): StackingClassifier definition
    │   ├── feature_quality.py              # NEW (v3.1): PCA/LDA/mutual-info/RFECV analysis
    │   ├── active_learning.py              # NEW (v3.1): uncertainty sampling, query-by-committee, self-training
    │   ├── model_loader.py
    │   └── feature_schema.py
    ├── rag/                         # hybrid BM25+dense retriever (v3.0), reranker/query-expansion/
    │                                 # GraphRAG/agent orchestrator (v3.1), LLM generator, kb/ (30 entries)
    ├── nlp/                          # NEW (v3.1): intent classification, symptom NER, dialogue state, urgency
    ├── xai/                          # NEW (v3.0): LIME (tabular) + global surrogate tree, alongside SHAP
    ├── evaluation/                   # NEW (v3.0/v3.1): McNemar's test, calibration, holdout runner,
    │                                 # stress testing, fairness audit, trust scores, PDP/ICE, SHAP interactions
    ├── mlops/                        # NEW (v3.0/v3.1): MLflow tracking, model registry, drift detection
    ├── sustainability/               # NEW (v3.1): energy/inference-cost benchmarking
    ├── scripts/
    │   ├── auto_label_from_filenames.py   # fills labels_template.csv from filenames
    │   ├── create_label_template.py
    │   ├── extract_features_dataset.py
    │   ├── retrain_model.py               # end-to-end retrain (extraction + training)
    │   └── cleanup_outputs.py             # manual/cron retention cleanup (also runs automatically now)
    ├── models/                      # svm_model.joblib, scaler.joblib, label_encoder.joblib,
    │                                 # shap_background.joblib, ensemble_model.joblib (NEW v3.0), metadata.json
    ├── data/                        # uploads/extracted/processed/outputs/logs + prsv_index.sqlite3;
    │                                 # training_features.csv + models/ tracked via DVC (NEW v3.1, see docs/DVC_SETUP.md)
    ├── labels_template.csv          # filled in (761/761 labeled) via auto_label_from_filenames.py
    ├── tests/                       # 109 pytest tests
    ├── requirements.txt              # base deps (includes lightweight v3.0/v3.1 additions)
    ├── requirements-optional.txt     # NEW (v3.0): sentence-transformers/faiss-cpu/mlflow/codecarbon
    └── .env.example
```


---

## Installation & Running It (Windows / VS Code)

### Prerequisites
- Python 3.10+ installed and on PATH (`python --version` in a terminal)
- VS Code with the Python extension (optional but recommended)

### Option A — one-click (`run.bat`)

Double-click `run.bat`, or from a terminal:

```bat
run.bat
```

This creates a virtual environment in `prsv_project\.venv`, installs `requirements.txt`, copies `.env.example` to `.env` if missing, warns if no trained model is present, and starts the server at `http://127.0.0.1:8000`.

### Option B — manual, inside VS Code

```powershell
cd prsv_project
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Open **http://127.0.0.1:8000** in your browser. In VS Code, open the `prsv_project` folder, select the `.venv` interpreter (bottom-right / `Ctrl+Shift+P` → "Python: Select Interpreter"), and you can run/debug `uvicorn` directly or use the built-in terminal for the commands above.

### Running the tests

```powershell
cd prsv_project
.venv\Scripts\activate
pytest -q
```

40 tests as of v2.9 (added coverage for the SQLite run index, the translation service's offline phrasebook, and the chatbot's greeting/Q&A handling).

### Troubleshooting: `numpy` (or another package) fails to build / "Unknown compiler(s)" during `pip install`

If `run.bat` or `pip install -r requirements.txt` fails while building a package from source (a wall of Meson/`vswhere.exe`/`cl`/`gcc`/Rust errors), it means pip couldn't find a prebuilt wheel for your exact Python version and fell back to compiling from C/Rust source — which needs a compiler you probably don't have installed.

`requirements.txt` uses minimum-version floors (`numpy>=1.26.4`, not `numpy==1.26.4`) specifically so pip always resolves to the *latest* release of each package, which is normally the one most likely to already have a wheel for a brand-new Python version. If you still hit this:

1. Delete the broken virtual environment: `rmdir /s /q prsv_project\.venv`
2. Check your Python version: `python --version`. If you're on a version released in the last few weeks, some packages may genuinely not have wheels yet — the safest fix is installing Python 3.12 or 3.13 (widely supported by every package this project uses) alongside your current Python, and pointing `run.bat`/the `venv` step at that interpreter instead (`py -3.12 -m venv .venv`).
3. Re-run `run.bat` (or the manual install steps above)

---

## Training / Retraining the Model

This zip already includes a trained model (`prsv_project/models/`), trained on the shipped `Original Images` dataset with labels auto-derived from filenames (`Healthy(n).jpg` → Healthy, `RingSpot(n).jpg` → Diseased) and cross-validated hyperparameters. To retrain from scratch (e.g. after adding your own images):

```powershell
cd prsv_project
.venv\Scripts\activate

# 1. Fill labels_template.csv. If you added new images, either edit the CSV by
#    hand or extend the filename convention and re-run:
python scripts\auto_label_from_filenames.py

# 2. Extract features + train (GridSearchCV over C/gamma, StratifiedKFold CV,
#    class_weight='balanced' for the 228/533 Healthy/Diseased imbalance):
python scripts\retrain_model.py
```

`retrain_model.py` writes `models/svm_model.joblib`, `scaler.joblib`, `label_encoder.joblib`, `metadata.json` (best hyperparameters + per-fold CV scores), `training_summary.json`, and `models/evaluation/` (confusion matrix, ROC, precision-recall plots). Restart the app afterward — `ml/infer_svm.py` will report `inference_mode: "trained_model"` instead of `heuristic_fallback`.

**On the shipped 761-image dataset with the v2.9 12-feature vector** (7 original + 5 GLCM/LBP texture features), this training run produced (held-out 20% test split): accuracy ≈ 0.95, F1 ≈ 0.92, ROC-AUC ≈ 0.98, best CV f1_macro ≈ 0.97. These numbers reflect the specific auto-labeled dataset and handcrafted feature set — treat them as a research baseline, not a validated clinical accuracy figure (see [Known Limitations](#known-limitations)).

---

## Deploying to Render (Free)

This repo includes `render.yaml` - a Render Blueprint that describes the whole deployment as code, so Render can set everything up automatically instead of you filling out a web form.

### Steps

1. Push this repo to GitHub (if you haven't already).
2. Go to [dashboard.render.com/blueprints](https://dashboard.render.com/blueprints) and sign up/log in - **no credit card required** for the free plan.
3. Click **New Blueprint Instance**, connect your GitHub account, and select this repo.
4. Render reads `render.yaml` automatically and shows you the `prsv-analyzer` web service it's about to create, using the free plan.
5. Click **Apply**. First deploy takes a few minutes (installing opencv/scikit-learn/etc. from scratch).
6. Once it's live, Render gives you a URL like `https://prsv-analyzer.onrender.com`.

### What to expect on the free tier

- **No persistent disk.** The SQLite run index and any photos/results from live visitors do **not** survive a restart. The trained model in `models/` still works every time, since it's committed to git and re-deploys with the code - only new activity done on the live site is lost when the free instance restarts. Fine for a public demo; not a substitute for a real deployment if you need permanent history.
- **Spins down after 15 minutes of no traffic**, and takes about a minute to wake back up on the next visit (you'll see Render's "waking up" page briefly). This is normal free-tier behavior, not a bug.
- **512MB RAM / 0.1 CPU.** This app's dependency stack (opencv, scikit-learn, shap, matplotlib, pandas) is on the heavier side for that allocation. It should run, but if you see slow responses or occasional crashes under load, that's the free tier's ceiling - upgrading to Render's cheapest paid instance ($7/mo Starter, 512MB/0.5 CPU) is the fix if that ever matters to you.
- **750 free instance-hours/month.** A single demo project run continuously would use ~730 hours in a 31-day month - close to the limit but should fit, especially since the service spins down when idle instead of running 24/7.

### CI (already wired up)

`.github/workflows/ci.yml` runs the full pytest suite on every push/PR to `main` via GitHub Actions - a genuine safety net that catches breakage before Render ever tries to deploy it. This runs automatically once the repo is on GitHub; no setup needed beyond the file already being there.

---

## Farmer Chatbot

A floating chat widget (bottom-right, on every page, desktop and mobile) gives farmers a conversational entry point to the same underlying system - it's the same web app, not a separate product.

**What it can do:**
- Answer questions about PRSV (symptoms, prevention, treatment, when to consult an extension officer) in English or 9 major Indian languages, grounded in `rag/kb/prsv_knowledge.json`
- Take a leaf photo directly in the chat and run it through the exact same analysis pipeline as the main upload page, then explain the result conversationally in the farmer's chosen language
- Accept voice input and read replies aloud, via the browser's built-in Web Speech API (no server-side speech model)

**How translation works:** typed/spoken text and generated replies are translated via `deep-translator` (free, no API key, hits a third-party translation backend over the internet). If that's unreachable, the widget falls back to a small pre-translated phrasebook for common moments (greeting, results, errors) so it still says something meaningful rather than silently defaulting to English with no explanation - see [Known Limitations](#known-limitations).

**Supported languages:** English, Hindi, Telugu, Tamil, Kannada, Malayalam, Bengali, Marathi, Gujarati, Punjabi.

**LLM call optional.** As of v3.0, the chatbot can use a real, grounded Claude API call (`rag/llm_generator.py`) when `ANTHROPIC_API_KEY` is set; without it, answers are composed from retrieved knowledge base chunks, consistent with how the rest of this app works (see [What This Actually Is](#what-this-actually-is-read-this-first)). See `ROADMAP_v3.md` for what's next on this path (RAFT, LoRA fine-tuning on a curated agricultural Q&A corpus).

---

## API Reference

Base path: `/api/analysis`

| Method | Path | Description |
|---|---|---|
| POST | `/single` | Analyze one uploaded image (synchronous) |
| POST | `/multiple` | Analyze multiple uploaded images (synchronous) |
| POST | `/multiple-async` | Same as `/multiple`, returns a `job_id` immediately - poll `/job/{job_id}` |
| POST | `/zip` | Analyze all valid images inside an uploaded ZIP (synchronous) |
| POST | `/zip-async` | Same as `/zip`, async job variant |
| POST | `/demo?limit=N` | Analyze N images from the configured demo dataset (synchronous) |
| POST | `/demo-async?limit=N` | Same as `/demo`, async job variant |
| GET | `/job/{job_id}` | Poll status/progress of an async batch job |
| GET | `/runs` | List past run folders + summary counts (filesystem scan) |
| GET | `/runs/search?q=&page=&page_size=&sort_by=&sort_dir=` | SQLite-backed search/filter/sort/pagination over runs |
| GET | `/run/{run_id}` | Get a run's `batch_summary.json` |
| GET | `/run/{run_id}/image/{image_id}` | Get one image's features/prediction/severity/RAG output |
| GET | `/run/{run_id}/download` | Download a zipped bundle of a run's outputs |
| GET | `/run/{run_id}/report.pdf` | Download a PDF summary report for the whole run |
| GET | `/run/{run_id}/image/{image_id}/report.pdf` | Download a PDF report for one image result |

Base path: `/api/chatbot`

| Method | Path | Description |
|---|---|---|
| GET | `/languages` | List supported languages (code, English name, native name, speech locale) |
| POST | `/message` | Send a text message `{text, language}`, get a translated, KB-grounded reply |
| POST | `/analyze-image?language=` | Attach a leaf photo, get a translated conversational diagnosis (reuses the main analysis pipeline) |

`run_id`/`image_id` path parameters are validated against the exact format the app itself generates (`run_YYYY_MM_DD_HH_MM_SS_xxxxxx`, `img_xxxxxxxxxx`) and re-checked against the output directory before any filesystem access, closing the path-traversal risk that existed in earlier versions of these endpoints.

An optional API key gate and a simple in-memory rate limiter now protect every `/api/*` route (see [Security Notes](#security-notes)) - both are disabled/generous by default so local research use is unaffected.

---

## Configuration

All configuration is environment-driven via `prsv_project/.env` (see `.env.example`). Key settings:

| Variable | Default | Notes |
|---|---|---|
| `DEBUG` | `false` | Never set `true` outside your own machine — exposes stack traces to any caller |
| `DEMO_DATASET_PATH` | `<repo_root>/Original Images` | Portable by default; override for a different dataset location |
| `OUTPUT_RETENTION_DAYS` | `30` | Used by `scripts/cleanup_outputs.py` to delete old `data/outputs/run_*` folders |
| `MAX_UPLOAD_SIZE_MB` / `MAX_ZIP_SIZE_MB` | `25` / `200` | Upload size limits |

### Cleaning up old run folders

`data/outputs/` grows by one folder per run with no built-in cleanup. Run periodically (Task Scheduler on Windows, cron elsewhere):

```powershell
cd prsv_project
.venv\Scripts\activate
python scripts\cleanup_outputs.py --dry-run   # preview what would be deleted
python scripts\cleanup_outputs.py             # actually delete runs older than OUTPUT_RETENTION_DAYS
```

---

## Known Limitations

- **Handcrafted features, still not fully lesion-localized.** All 12 features (7 color/edge + 5 texture) are per-image or per-region statistics, not per-lesion. GLCM/LBP (added in v2.9) capture texture irregularity better than color alone, but this still isn't full lesion segmentation/counting.
- **Labels are filename-derived, not expert-annotated.** `auto_label_from_filenames.py` trusts whoever originally sorted the images into `Healthy(n)` / `RingSpot(n)`. Spot-check before treating the model as ground truth.
- **Severity has no ground truth.** It's an unsupervised weighted formula, not a regression trained against expert-annotated infection percentages — treat the number as indicative, not precise.
- **SHAP contributions are an approximation.** `nsamples=200` in `ml/shap_explainer.py` trades exactness for speed (~1s/image) - treat the bars as directionally informative, not exact Shapley values.
- **RAG (both the main pipeline and the chatbot) is retrieval + templates/composition, not LLM generation.** Explanations for similar cases will read similarly; see `rag/generator.py` / `app/services/chatbot_service.py` if you want to wire in an actual LLM call later.
- **Chatbot translation depends on outbound internet access.** `deep-translator` calls a free third-party translation backend with no API key; if unreachable, the chatbot falls back to English plus a small pre-translated phrasebook for common moments (greeting, results, errors) rather than failing outright - but free-form Q&A answers will show in English with a note when translation is down.
- **Chatbot voice support depends on the browser.** Voice input/output uses the Web Speech API, well-supported in Chrome/Edge, unsupported in Firefox, partial in Safari. The mic button is disabled with a tooltip when unsupported; text and photo chat always work.
- **Async batch jobs are in-process, single-worker.** The background job queue (`app/services/job_queue.py`) is a single `ThreadPoolExecutor` worker by design (see the file's docstring for the reasoning) - batch jobs run one at a time, not in parallel, and job state doesn't survive an app restart.

---

## Security Notes

- `DEBUG` now defaults to `false`. Keep it that way outside local development.
- Path-traversal on the `run_id`/`image_id` lookup endpoints is closed (strict format whitelist + containment check against the resolved output directory).
- An optional API key gate and a simple in-memory rate limiter (`app/middleware.py`) now protect every `/api/*` route. Both are effectively off by default (`API_KEY` blank, generous rate limit) so local research use is unaffected - set `API_KEY` in `.env` and require callers to send it back as the `X-API-Key` header before deploying beyond your own machine. The rate limiter is per-process/per-IP and resets on restart - fine for a single-machine deployment, not a substitute for a real gateway at real scale.
- The chatbot's translation calls go out to a third-party translation service over the internet - if you're deploying somewhere with restricted egress, translation will silently fall back to the English/phrasebook path rather than fail the request (see [Known Limitations](#known-limitations)).

---

## Future Scope

v3.0 and v3.1 together shipped the large majority of this project's original
upgrade wishlist (see `CHANGELOG_v3.0.md` and `CHANGELOG_v3.1.md`) - hybrid
retrieval, real LLM generation, reranking, GraphRAG, agentic routing,
chatbot NLP (intent/symptom/urgency), extended texture/color-space/
segmentation feature engineering, a stacked ensemble with statistical
comparison, active learning, a model registry, drift detection, stress
testing, fairness auditing, trust scores, PDP/ICE, SHAP interactions,
energy benchmarking, and real DVC tracking.

For the full, honestly-scoped list of what's *still* not implemented and
why (grouped by "needs a dataset that doesn't exist yet" vs. "needs
infrastructure this project doesn't have" - after v3.1 there is no
remaining "just not built yet" category), see
**[`ROADMAP_v3.md`](ROADMAP_v3.md)** at the repo root. Remaining highlights:

- A CNN/transfer-learning model is a natural next step technically, but is **explicitly out of scope for this project** per current direction - not planned
- Trained lesion segmentation (SAM/U-Net) and trained severity regression, both blocked on labeled data this project doesn't have
- AI4Bharat's IndicTrans2/Indic-TTS as a higher-quality, India-specific alternative to the current free translation service
- Postgres instead of SQLite if this ever needs true multi-instance/concurrent-write deployment
- A real distributed task queue (Celery/Redis) if batch processing ever needs to scale beyond one machine
- Federated learning, on-device inference, and LoRA fine-tuning, all blocked on infrastructure (multi-device test fleet, mobile app target, GPU training) this project doesn't have

---

## Changelog

### v3.1.0

See **[`CHANGELOG_v3.1.md`](CHANGELOG_v3.1.md)** for the full list. Summary:
cross-encoder reranking, query expansion, structured/constrained LLM
output, Corrective RAG, GraphRAG, agentic message routing; real intent
classification, symptom extraction, and urgency detection wired into the
chatbot; extended texture/color-space/illumination/segmentation feature
engineering; PCA/mutual-information/RFECV feature-quality analysis; active
learning; a model registry with rollback; drift detection; stress testing,
fairness auditing, trust scores, PDP/ICE, and SHAP interaction values;
energy benchmarking; and genuinely initialized DVC tracking. 109 tests
passing (up from 65). See `ROADMAP_v3.md` for what's still deliberately not
done and why - after v3.1, everything remaining is blocked on a dataset or
infrastructure this project doesn't have, not on undone engineering work.

### v3.0.0

See **[`CHANGELOG_v3.0.md`](CHANGELOG_v3.0.md)** for the full, file-by-file
list. Summary: hybrid BM25+dense RAG retrieval with automatic degradation,
real grounded LLM generation with template fallback, RAGAS-lite evaluation,
LIME + global surrogate tree explainability, calibrated + stacked-ensemble
classification with statistical comparison (McNemar's test), lesion-ratio
severity with a bootstrap uncertainty interval, calibration/Brier
evaluation, an independent-source holdout runner, MLflow tracking, and a
Redis-backed (memory-fallback) cache. All additive to v2.9 - the full
pre-existing test suite (55 tests) passes unmodified, plus 10 new test files
for the new modules.


**Deployment**
- Added `render.yaml` - a Render Blueprint for one-click, no-credit-card-required free deployment (Docker support was considered and explicitly removed per project direction; Render's native Python runtime is used instead)
- Added `.github/workflows/ci.yml` - GitHub Actions CI that runs the full test suite on every push/PR to `main`

**Multilingual farmer chatbot (web widget, no separate app)**
- Floating chat widget on every page, works on both desktop and mobile browsers
- Text chat in English + 9 major Indian languages (Hindi, Telugu, Tamil, Kannada, Malayalam, Bengali, Marathi, Gujarati, Punjabi), grounded in the same knowledge base the rest of the app uses (no external LLM call - retrieval + composition, consistent with how `rag/` already works)
- Voice input/output via the browser's built-in Web Speech API (no server-side speech model needed) - gracefully degrades to text-only in browsers without support (e.g. Firefox)
- Translation via `deep-translator` (free, no API key) with an offline phrasebook fallback for common moments (greeting, results, errors) so the widget still communicates something meaningful if live translation is unreachable
- Photo-in-chat diagnosis: attach a leaf photo directly in the chat and get a conversational, translated explanation - reuses the exact same analysis pipeline as the main upload page
- Knowledge base expanded from 10 to 30 entries with farmer-facing content: treatment steps, prevention, when to consult an extension officer, photo-quality tips, and how to use the tool itself

**Frontend overhaul**
- Drag-and-drop upload with image preview thumbnails, client-side validation, and camera capture on mobile browsers
- Real upload/processing progress (replacing the old fake timed animation) via XHR + a new async job queue, with live percentage and "processing N of M" status
- Interactive Chart.js charts (prediction/severity distribution, SHAP feature contributions) replacing static matplotlib PNGs on the run and image detail pages
- Before/after comparison slider (original vs. severity overlay) and a click-to-zoom lightbox on result images
- Dark mode toggle (persisted), responsive hamburger nav for mobile
- Client-side pagination and instant filter-as-you-type on the Runs/Batch pages
- Downloadable PDF report (single image or whole run) via a new reportlab-based report generator
- Cleaned up three previously dead, empty JS files (`analyze.js`, `dashboard.js`, `upload.js` were 0 bytes and unused) - now populated with real functionality

**Backend/ML (CNN intentionally excluded per project scope)**
- Added GLCM + LBP texture features (5 new features on top of the original 7) - ring-spot lesions have a distinct local texture that global color/edge averages miss; model retrained on all 761 images with the expanded 12-feature vector
- SHAP-based explainability (`ml/shap_explainer.py`) - per-feature contribution toward each prediction, with a k-means-summarized background sample for fast inference-time computation (~1s/image after a one-time warm-up at startup)
- SQLite run index (`app/services/run_store.py`) - additive alongside the existing JSON files, enables fast search/pagination without scanning the filesystem
- Lightweight in-process background job queue (`app/services/job_queue.py`) for batch/ZIP/demo uploads - deliberately not Celery/Redis, since this app targets a single-machine deployment; new `-async` endpoint variants return a `job_id` immediately and the frontend polls for progress
- Optional API key gate + simple in-memory rate limiting on `/api/*` routes (`API_KEY`, `RATE_LIMIT_REQUESTS`, `RATE_LIMIT_WINDOW_SECONDS` in `.env`) - disabled by default for local use
- Automatic scheduled cleanup (runs every 24h in the background, alongside the existing manual `scripts/cleanup_outputs.py`)

### v2.8.0
- **Fixed:** `pip install -r requirements.txt` failed with build-from-source errors (`Unknown compiler(s): [cl, gcc, ...]`) on newer Python versions. First attempt pinned exact newer versions (`numpy==2.1.3` etc.), which fixed it for Python ≤3.13 but still failed on Python 3.14 (`numpy` and `pydantic-core` had no 3.14 wheels yet at those exact pinned versions). Switched `requirements.txt` from exact pins to minimum-version floors (`numpy>=1.26.4`, `opencv-python-headless>=4.10.0.84`, etc.) so pip always resolves to whatever's newest — which is normally the version most likely to already have a wheel for a brand-new Python release. Verified against a clean install of everything pip currently resolves to (numpy 2.5.1, scikit-learn 1.9.0, pandas 3.0.3, opencv 5.0.0, pydantic 2.13.4, fastapi 0.139.0 at time of writing) — full test suite passes with 0 code changes needed. The shipped model was retrained against these exact resolved versions to avoid `InconsistentVersionWarning` at load time.
- Trained a real SVM model on all 761 shipped images (auto-labeled from filenames), with GridSearchCV + StratifiedKFold cross-validation and `class_weight='balanced'` — replaces the previous untrained heuristic-fallback-only state.
- **Fixed:** single-image uploads (both the web form and `POST /api/analysis/single`) now go through the same batch pipeline as multi-image runs, so `batch_summary.json` is always written. Previously, single-image runs had no summary file, causing `/run/{run_id}`, `/api/analysis/run/{run_id}`, and the "Open Run" links on the Home/Results pages to 404.
- **Fixed:** the single-image redirect used to derive `run_id` from a `Path(...).parents[N]` index that pointed at the literal `"images"` folder name instead of the run folder, sending users to a broken `/run/images/image/{image_id}` URL. Root-caused and fixed by removing that logic entirely (see above).
- Severity is now suppressed to the "Healthy" band whenever the classifier predicts Healthy, so the UI can no longer show a contradictory state like "Healthy – Mild severity".
- Closed a path-traversal risk on the `run_id`/`image_id` lookup endpoints (strict format whitelist + containment check).
- `DEBUG` now defaults to `false`; the hardcoded Windows demo-dataset path is now a portable relative default.
- `/api/analysis/runs` no longer leaks the server's absolute filesystem path in its response.
- Converted the deprecated `@app.on_event("startup")` hook to FastAPI's `lifespan` context manager.
- Added `scripts/auto_label_from_filenames.py`, fixed `scripts/retrain_model.py` (it previously extracted features but never actually trained anything), and added `scripts/cleanup_outputs.py` for output-folder retention.
- Rewrote this README to match the actual FastAPI/joblib architecture (an earlier version described a Flask app that never existed in this codebase).

---

## License

MIT License — see `LICENSE` if present, or treat as MIT by default for this research project.

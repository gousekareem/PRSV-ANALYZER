# PRSV Analyzer — Research Roadmap (post-v3.1)

This document exists because the original upgrade request enumerated roughly
150 research directions across 14 sections. v3.0 and v3.1 together implement
a large, real, working subset (see `CHANGELOG_v3.0.md` and
`CHANGELOG_v3.1.md`). This file is the honest account of everything that
remains, and why.

v3.1 closed out essentially all of the "not blocked, just not built yet"
category from the original roadmap. What's left below is now almost
entirely genuinely blocked on data or infrastructure this project doesn't
have — not on remaining engineering effort.

---

## Category A — Still needs a labeled dataset that doesn't exist yet

Unchanged from before v3.1; nothing in this category can be built without
data collection that hasn't happened.

- **Trained lesion segmentation** (SAM/SAM2 fine-tuning, U-Net/U-Net++,
  DeepLabV3+, Mask R-CNN/YOLOv8-seg instance counting) — needs
  manually-annotated lesion masks. `image_processing/segmentation_classical.py`
  (v3.1) adds real classical alternatives (GrabCut, Watershed, SLIC) that
  need no training data, as the closest available step in this direction.
- **Trained severity regression + genuine conformal calibration** — needs
  expert-annotated severity scores. `image_processing/severity_v2.py`'s
  `calibrate_from_residuals()` (v3.0) is the real conformal machinery,
  ready the moment such labels exist.
- **Multi-annotator severity labeling + inter-rater agreement** — needs
  multiple domain experts labeling the same images; prerequisite to the
  item above.
- **Independent-source holdout evaluation** — `evaluation/holdout_eval.py`
  (v3.0) is real, working infrastructure with nothing to evaluate against
  yet; confirmed in v3.1 testing to report this honestly rather than
  fabricate a result (see its "no_independent_source_data" status).
- **Cross-dataset benchmarking against PlantVillage/PlantDoc** — needs
  downloading and reformatting those datasets; not started.
- **GAN-based synthetic augmentation (StyleGAN2)** — flagged in the
  original wishlist itself as crossing the deep-learning boundary excluded
  from this project.
- **Crowdsourced labeling loop** — needs live production traffic and a
  consent/outcome-confirmation flow that doesn't exist without a real
  deployment.
- **Fairness/bias audit with real subgroup data** — `evaluation/fairness_audit.py`
  (v3.1) is real, tested, working code; it needs a subgroup/source column
  (e.g. camera/session) in the feature manifest, which the current dataset
  doesn't track per-image. One column addition to
  `scripts/prepare_manifest.py` away from being exercised for real.

## Category B — Still needs infrastructure this project doesn't have

- **Federated learning (Flower)** — needs multiple real/simulated
  farmer-device endpoints.
- **On-device inference (TFLite/ONNX Mobile)** — needs a mobile app target;
  this is a server-side FastAPI app.
- **LoRA/QLoRA fine-tuning, RAFT** — needs GPU training infra and a curated
  Q&A corpus. `rag/llm_generator.py`'s grounded prompting (v3.0) and
  `rag/structured_output.py`'s constrained generation (v3.1) are the
  zero-training-cost steps in this direction already shipped.
- **India-specific LLMs / AI4Bharat's IndicTrans2/IndicWhisper/Indic-TTS** —
  needs hosting infra these models don't have as a simple API.
  `rag/embeddings_retriever.py` (v3.0) already uses a general multilingual
  embedding model as an interim step.
- **Drone/satellite NDVI, IoT sensor fusion** — needs real sensor/imagery
  data feeds.
- **Weather API fusion for risk forecasting** — needs a validated
  weather-to-PRSV-risk correlation model, a research question in its own
  right, not a data-pipe problem.
- **Distributed task queue (Celery), GraphQL, WebSocket streaming, Postgres
  migration** — genuine engineering tasks, deliberately out of scope for
  this single-machine-deployment-targeted project (see README's existing
  rationale for SQLite-over-Postgres, in-process-queue-over-Celery).

## Category C — Closed out in v3.1

Everything below was listed as "not blocked, just not prioritized" as of
v3.0 and is now real, tested, working code as of v3.1:

- Cross-encoder reranking (`rag/reranker.py`)
- Query expansion/rewriting (`rag/query_expansion.py`)
- Structured/constrained generation (`rag/structured_output.py`)
- Corrective RAG relevance grading (`rag/corrective_rag.py`)
- GraphRAG multi-hop traversal (`rag/knowledge_graph.py`)
- Agentic RAG orchestration (`rag/agent_orchestrator.py`)
- Intent classification (`nlp/intent_classifier.py`) — wired into `chatbot_service.py`
- Symptom/plant-part extraction (`nlp/symptom_ner.py`) — wired into `chatbot_service.py`
- Multi-turn dialogue state + coreference resolution (`nlp/dialogue_state.py`)
- Urgency/distress detection (`nlp/urgency_detection.py`) — wired into `chatbot_service.py`
- PCA/LDA/mutual-information feature-quality analysis, RFECV (`ml/feature_quality.py`)
- Uncertainty sampling, query-by-committee, self-training (`ml/active_learning.py`)
- Model registry with rollback + champion/challenger (`mlops/model_registry.py`)
- Drift detection via PSI/KS-test (`mlops/drift_detection.py`)
- Adversarial/stress testing (`evaluation/stress_testing.py`)
- Fairness/bias auditing machinery (`evaluation/fairness_audit.py`)
- Trust scores (`evaluation/trust_scores.py`)
- PDP/ICE curves (`evaluation/pdp_ice.py`)
- SHAP interaction values (`evaluation/shap_interactions.py`)
- Multi-radius LBP, Gabor filters, Tamura texture, wavelet energy, LPQ (`image_processing/texture_extended.py`)
- Lab/YCbCr/HSI color spaces, vegetation indices, color moments (`image_processing/color_spaces_extended.py`)
- Retinex illumination normalization, shadow detection, gray-world white balance (`image_processing/illumination_normalization.py`)
- GrabCut/Watershed/SLIC segmentation alternatives (`image_processing/segmentation_classical.py`)
- Data Version Control — genuinely initialized (`git log`, `.dvc/`, tracked `training_features.csv` + `models/`; see `docs/DVC_SETUP.md`)
- Energy/inference-cost benchmarking (`sustainability/energy_benchmark.py`)

Each of these is real code with passing tests (109 tests total across v3.0+v3.1), not a placeholder — see `CHANGELOG_v3.1.md` for the full file-by-file list, including the two real bugs (a model-registry timestamp collision, a SHAP array-shape mismatch) that the test suite caught before release.

## What's genuinely left

After v3.1, essentially everything remaining is in Category A or B above —
blocked on a dataset or infrastructure this project doesn't have, not on
undone engineering work. The honest summary: **the engineering backlog from
the original wishlist is closed**; what's left is a data-collection and
infrastructure-provisioning roadmap, not a coding one.

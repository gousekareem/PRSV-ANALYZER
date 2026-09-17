# Changelog — v3.0.0

Upgrade from v2.9. See `ROADMAP_v3.md` for everything from the original
14-section wishlist that is *not* in this release, and why.

## RAG + LLM
- **Hybrid retrieval**: `rag/hybrid_retriever.py` combines BM25
  (`rank_bm25`) and dense multilingual embeddings
  (`rag/embeddings_retriever.py`, `sentence-transformers` +
  FAISS/numpy) via Reciprocal Rank Fusion. Automatically degrades to
  BM25-only, dense-only, or the original TF-IDF retriever depending on
  which optional packages are installed — never a regression.
- **Real LLM generation**: `rag/llm_generator.py` calls the Claude API with
  a strict grounded-context prompt when `ANTHROPIC_API_KEY` is set; falls
  back to the original deterministic template generator otherwise.
- **RAG evaluation**: `rag/rag_eval.py` implements RAGAS-lite faithfulness,
  answer relevance, context precision, and context recall, using embedding
  similarity (or token-overlap fallback) instead of an LLM judge, so it runs
  free and offline.
- Wired into `app/services/rag_service.py` and `chatbot_service.py`, both
  now cached via the new `app/services/cache_service.py`.

## Explainability
- `xai/lime_explainer.py`: tabular LIME as a second, model-agnostic local
  explanation alongside the existing SHAP explainer (documented honestly:
  this is tabular LIME, not superpixel LIME, because the classifier
  consumes engineered features, not raw pixels).
- `xai/surrogate_tree.py`: global decision-tree distillation of the
  classifier's overall decision boundary, with a fidelity score.

## Classical ML
- `ml/calibration.py`: `CalibratedClassifierCV` (Platt/isotonic) replacing
  the in-sample-fit `SVC(probability=True)` pattern.
- `ml/ensemble.py` + `ml/train_ensemble.py`: real `StackingClassifier`
  (SVM + Random Forest + Gradient Boosting → Logistic Regression
  meta-learner), trained and saved separately from the production SVM so
  the two can be statistically compared.

## Severity
- `image_processing/severity_v2.py`: direct lesion-area-ratio severity
  (symptom pixels / leaf pixels) plus a bootstrap sensitivity interval.
  Includes the real split-conformal calibration function
  (`calibrate_from_residuals`), ready for use once expert-annotated
  severity ground truth exists.

## Evaluation rigor
- `evaluation/statistical_tests.py`: McNemar's test for comparing two
  classifiers on the same test set, plus calibration curves and Brier score.
- `evaluation/holdout_eval.py`: independent-source holdout evaluation
  runner (real infrastructure; honestly reports "no data yet" until a
  second-source dataset is collected).

## MLOps
- `mlops/experiment_tracking.py`: MLflow wrapper (no-op when not
  configured/installed) used by `ml/train_ensemble.py`.

## Infrastructure
- `app/services/cache_service.py`: Redis-backed cache with an in-process
  fallback, used for repeated chatbot/RAG queries.
- New config flags in `app/config.py` / `.env`: `RAG_USE_HYBRID_RETRIEVAL`,
  `RAG_USE_LLM_GENERATION`, `RAG_CANDIDATE_POOL_SIZE`, `ANTHROPIC_API_KEY`,
  `REDIS_URL`, `CACHE_TTL_SECONDS`, `MLFLOW_TRACKING_URI`,
  `MLFLOW_EXPERIMENT_NAME`.
- `requirements.txt` split: lightweight v3.0 additions stay in the base
  file; `sentence-transformers`/`faiss-cpu`/`mlflow` moved to
  `requirements-optional.txt` since they pull in PyTorch.
- CI (`.github/workflows/ci.yml`) now runs a required `test-base` job
  (lightweight deps) and a best-effort `test-full` job (with the optional
  heavy deps installed).

## Schema changes
- `ImageResult` gains two new optional fields: `lime_contributions` and
  `lesion_ratio_severity`. Both default to `None`/absent, so existing
  serialized runs and API consumers are unaffected.

## Testing
- 10 new test files covering every new module
  (`test_hybrid_retriever.py`, `test_llm_generator.py`, `test_rag_eval.py`,
  `test_cache_service.py`, `test_calibration.py`, `test_ensemble.py`,
  `test_surrogate_tree.py`, `test_statistical_tests.py`,
  `test_severity_v2.py`, `test_lime_explainer.py`), verified to pass both
  with and without the optional heavy dependencies installed.
- Full existing test suite (55 pre-existing tests) still passes unmodified,
  confirming v3.0 is additive, not a rewrite.

## Fixed
- `ImageResult.lesion_ratio_severity` schema bug caught by the test suite
  during this release (typed as `Dict[str, float]`, which rejected the
  `interval_method` string field) — fixed to `Dict[str, Any]` before
  release.

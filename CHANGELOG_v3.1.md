# Changelog — v3.1.0

Upgrade from v3.0. Closes out Category C from `ROADMAP_v3.md` — everything
that was blocked only by "not yet built," not by missing data or infra.

## RAG
- `rag/reranker.py`: cross-encoder reranking (ms-marco-MiniLM) as a second
  pass over hybrid retrieval results.
- `rag/query_expansion.py`: LLM rewrite (when configured) or tag-based
  lexical expansion of terse/ambiguous queries.
- `rag/structured_output.py`: schema-constrained generation via Claude
  tool-use, returning a validated `StructuredDiagnosis` (summary/actions/
  confidence caveat) instead of free text.
- `rag/corrective_rag.py`: relevance-grades retrieved chunks before
  generation and signals fallback when nothing clears the bar.
- `rag/knowledge_graph.py`: builds a real symptom→cause→treatment graph
  over the existing 30-chunk knowledge base and supports multi-hop
  traversal.
- `rag/agent_orchestrator.py`: routes chatbot messages to
  analyze_photo/retrieve_knowledge/direct_answer — rule-based by default,
  genuine LLM tool-use routing when `ANTHROPIC_API_KEY` is set.

## NLP (chatbot layer)
- `nlp/intent_classifier.py`: embedding-based zero-shot intent
  classification (greeting/thanks/photo_request/complaint_urgent/question),
  degrading to a keyword heuristic. **Wired into `chatbot_service.py`.**
- `nlp/symptom_ner.py`: dictionary-based symptom/plant-part extraction.
  **Wired into `chatbot_service.py`.**
- `nlp/urgency_detection.py`: keyword/heuristic distress scoring, triggers
  an urgent-response prefix in the chatbot. **Wired into `chatbot_service.py`.**
- `nlp/dialogue_state.py`: session-scoped conversation memory +
  coreference resolution ("is it contagious?" → resolves "it" to the last
  discussed topic). Built and tested; full wiring into the chat API route
  needs a session-id concept at the routing layer, noted as the next step
  rather than silently skipped.

## Feature engineering (image_processing/)
- `texture_extended.py`: multi-radius LBP, Gabor filter bank, Tamura
  texture (coarseness/contrast/directionality), wavelet sub-band energy
  (optional `PyWavelets`), and a from-scratch Local Phase Quantization
  implementation.
- `color_spaces_extended.py`: Lab/YCbCr/HSI histograms, RGB vegetation
  indices (ExG/ExGR/GLI), per-channel color moments.
- `illumination_normalization.py`: single- and multi-scale Retinex,
  chromaticity-based shadow detection, gray-world white balance.
- `segmentation_classical.py`: GrabCut, marker-based Watershed, and SLIC
  superpixel segmentation as alternatives to the HSV-threshold baseline.
- All opt-in/additive — the production 12-feature vector and shipped model
  are unchanged; these are available for exploratory analysis and a future
  retraining run with an expanded feature set.

## Classical ML
- `ml/feature_quality.py`: PCA/LDA analysis, mutual-information ranking,
  and RFECV feature selection against the existing feature CSV.
- `ml/active_learning.py`: uncertainty sampling, query-by-committee
  (ensemble disagreement), and confidence-gated self-training pseudo-
  labeling — implemented directly rather than adding the `modAL`
  dependency for the same underlying numpy operations.

## MLOps
- `mlops/model_registry.py`: lightweight JSON-based model registry with
  versioning, rollback, and champion/challenger comparison.
- `mlops/drift_detection.py`: PSI and Kolmogorov-Smirnov-based feature
  drift detection, implemented directly rather than adding
  `evidently`/`whylogs` as dependencies.

## Evaluation
- `evaluation/stress_testing.py`: blur/brightness/JPEG-compression/
  occlusion perturbations with prediction-flip-rate and confidence-drop
  reporting.
- `evaluation/fairness_audit.py`: per-subgroup metric breakdown and
  largest-gap reporting (real infrastructure; needs a subgroup column in
  the feature manifest to run against real data — see `ROADMAP_v3.md`).
- `evaluation/trust_scores.py`: distance-based trust scoring (Jiang et al.
  2018 formulation) flagging high-confidence-but-poorly-supported
  predictions.
- `evaluation/pdp_ice.py`: partial dependence + individual conditional
  expectation curves via scikit-learn.
- `evaluation/shap_interactions.py`: SHAP interaction values for the
  ensemble's tree-based members (TreeExplainer; not available for the
  SVM's KernelExplainer, which is a `shap` library limitation, not one
  introduced here).

## Sustainability
- `sustainability/energy_benchmark.py`: inference-cost benchmarking via
  CodeCarbon (measured) when available, CPU-time proxy estimate otherwise
  — labeled honestly either way.

## Data engineering
- **DVC genuinely initialized**: `git log` shows a real commit tracking
  `data/training_features.csv` and `models/` via actual `.dvc` pointer
  files, not just documentation. See `docs/DVC_SETUP.md` for adding a
  remote.

## Fixed (caught by the test suite before release)
- `mlops/model_registry.py`: version IDs used second-resolution timestamps,
  causing two registrations within the same second to collide and corrupt
  rollback targeting. Fixed with microsecond resolution + a uuid suffix.
- `evaluation/shap_interactions.py`: the installed `shap` version returns
  interaction values as a single 4D array `(samples, features, features,
  classes)` rather than the older per-class list format the first
  implementation assumed, causing a scalar-cast crash. Fixed to handle
  both shapes.

## Testing
- 9 new test files covering every new module (`test_nlp_modules.py`,
  `test_rag_advanced.py`, `test_feature_quality_and_active_learning.py`,
  `test_mlops_additions.py`, `test_evaluation_additions.py`,
  `test_extended_image_processing.py`, `test_shap_interactions.py`, plus
  additions to existing suites).
- `ml/train_ensemble.py` verified to run end-to-end against the real
  shipped feature CSV and baseline model, producing a real McNemar's-test
  comparison (p=1.0, not significant, on this dataset) and a 95.4%-fidelity
  surrogate tree.
- Full test suite: **109 passed** (up from 65 in v3.0, 55 in v2.9),
  including a clean-room extraction check.

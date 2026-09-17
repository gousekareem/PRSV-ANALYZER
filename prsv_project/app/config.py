from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """
    Central application configuration for the PRSV research system.
    All paths are resolved in a Windows-safe manner using pathlib.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = Field(default="PRSV Research Diagnostic System", alias="APP_NAME")
    app_version: str = Field(default="3.1.0", alias="APP_VERSION")
    # SECURITY: debug defaults to False. Never ship debug=True to anything but a
    # local dev machine - Starlette/FastAPI will happily return stack traces to
    # any caller when this is on. Opt in explicitly via .env (DEBUG=true).
    debug: bool = Field(default=False, alias="DEBUG")

    # Portable default: <repo_root>/Original Images (the folder that ships next
    # to prsv_project/ in this zip). Override via .env (DEMO_DATASET_PATH=...)
    # for any other location. No more hardcoded Windows drive letters.
    demo_dataset_path: Path = Field(
        default=ROOT_DIR.parent / "Original Images",
        alias="DEMO_DATASET_PATH",
    )

    max_upload_size_mb: int = Field(default=25, alias="MAX_UPLOAD_SIZE_MB")
    max_zip_size_mb: int = Field(default=200, alias="MAX_ZIP_SIZE_MB")

    # Retention policy for data/outputs/ run folders. Runs older than this many
    # days are eligible for deletion by scripts/cleanup_outputs.py. Set to 0 to
    # disable age-based cleanup entirely.
    output_retention_days: int = Field(default=30, alias="OUTPUT_RETENTION_DAYS")

    # Optional API key gate for /api/* routes. Blank (default) = disabled, so
    # local/research use is unaffected. Set API_KEY in .env to require callers
    # to send it back as the X-API-Key header before deploying beyond your
    # own machine.
    api_key: str = Field(default="", alias="API_KEY")

    # Simple in-memory sliding-window rate limit applied to /api/* routes.
    rate_limit_requests: int = Field(default=60, alias="RATE_LIMIT_REQUESTS")
    rate_limit_window_seconds: int = Field(default=60, alias="RATE_LIMIT_WINDOW_SECONDS")

    image_width: int = Field(default=128, alias="IMAGE_WIDTH")
    image_height: int = Field(default=128, alias="IMAGE_HEIGHT")

    rag_top_k: int = Field(default=3, alias="RAG_TOP_K")

    # v3.0 RAG upgrade path. All default to True/on but are self-degrading:
    # HybridRetriever automatically falls back to BM25-only, dense-only, or
    # the original TF-IDF retriever depending on which optional packages are
    # actually installed (see rag/hybrid_retriever.py). Setting either flag to
    # False forces the v2.9 TF-IDF-only behavior even if the packages ARE
    # installed, useful for A/B comparison or reverting without a code change.
    rag_use_hybrid_retrieval: bool = Field(default=True, alias="RAG_USE_HYBRID_RETRIEVAL")
    rag_use_llm_generation: bool = Field(default=True, alias="RAG_USE_LLM_GENERATION")
    rag_candidate_pool_size: int = Field(default=15, alias="RAG_CANDIDATE_POOL_SIZE")

    # Optional Redis cache for repeated chatbot/RAG queries. Blank URL (the
    # default) disables Redis entirely and app/services/cache_service.py
    # transparently falls back to an in-process dict cache, so nothing breaks
    # on a machine without Redis installed.
    redis_url: str = Field(default="", alias="REDIS_URL")
    cache_ttl_seconds: int = Field(default=3600, alias="CACHE_TTL_SECONDS")

    # MLflow experiment tracking for model training runs. Blank (default)
    # disables it; scripts still run and print results to stdout/JSON either
    # way (see ml/train_svm.py / ml/train_ensemble.py + mlops/experiment_tracking.py).
    mlflow_tracking_uri: str = Field(default="", alias="MLFLOW_TRACKING_URI")
    mlflow_experiment_name: str = Field(default="prsv-analyzer", alias="MLFLOW_EXPERIMENT_NAME")

    enable_denoising: bool = Field(default=True, alias="ENABLE_DENOISING")
    enable_clahe: bool = Field(default=True, alias="ENABLE_CLAHE")
    enable_debug_visuals: bool = Field(default=True, alias="ENABLE_DEBUG_VISUALS")

    @property
    def image_size(self) -> Tuple[int, int]:
        return (self.image_width, self.image_height)

    @property
    def data_dir(self) -> Path:
        return ROOT_DIR / "data"

    @property
    def upload_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def extracted_dir(self) -> Path:
        return self.data_dir / "extracted"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def output_dir(self) -> Path:
        return self.data_dir / "outputs"

    @property
    def temp_dir(self) -> Path:
        return self.data_dir / "temp"

    @property
    def log_dir(self) -> Path:
        return self.data_dir / "logs"

    @property
    def models_dir(self) -> Path:
        return ROOT_DIR / "models"

    @property
    def reports_dir(self) -> Path:
        return ROOT_DIR / "reports"

    @property
    def kb_path(self) -> Path:
        return ROOT_DIR / "rag" / "kb" / "prsv_knowledge.json"

    @property
    def model_path(self) -> Path:
        return self.models_dir / "svm_model.joblib"

    @property
    def scaler_path(self) -> Path:
        return self.models_dir / "scaler.joblib"

    @property
    def label_encoder_path(self) -> Path:
        return self.models_dir / "label_encoder.joblib"

    @property
    def model_metadata_path(self) -> Path:
        return self.models_dir / "metadata.json"

    @property
    def shap_background_path(self) -> Path:
        return self.models_dir / "shap_background.joblib"

    @property
    def ensemble_model_path(self) -> Path:
        return self.models_dir / "ensemble_model.joblib"

    @property
    def calibrated_model_path(self) -> Path:
        return self.models_dir / "svm_model_calibrated.joblib"

    @property
    def allowed_extensions(self) -> List[str]:
        return [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

    @property
    def severity_thresholds(self) -> Dict[str, Tuple[float, float]]:
        return {
            "Healthy": (0.0, 0.0),
            "Very mild": (0.1, 5.0),
            "Mild to moderate": (5.1, 25.0),
            "Moderate": (25.1, 50.0),
            "Moderate to severe": (50.1, 75.0),
            "Severe": (75.1, 100.0),
        }

    @property
    def severity_weights(self) -> Dict[str, float]:
        return {
            "inverse_green_ratio": 0.30,
            "edge_density": 0.20,
            "entropy": 0.15,
            "abnormal_color_score": 0.20,
            "symptom_region_ratio": 0.15,
        }


settings = Settings()
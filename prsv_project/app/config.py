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
    app_version: str = Field(default="1.0.0", alias="APP_VERSION")
    debug: bool = Field(default=True, alias="DEBUG")

    demo_dataset_path: Path = Field(
        default=Path(r"D:\PRSV PROJECT\original images"),
        alias="DEMO_DATASET_PATH",
    )

    max_upload_size_mb: int = Field(default=25, alias="MAX_UPLOAD_SIZE_MB")
    max_zip_size_mb: int = Field(default=200, alias="MAX_ZIP_SIZE_MB")

    image_width: int = Field(default=128, alias="IMAGE_WIDTH")
    image_height: int = Field(default=128, alias="IMAGE_HEIGHT")

    rag_top_k: int = Field(default=3, alias="RAG_TOP_K")

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
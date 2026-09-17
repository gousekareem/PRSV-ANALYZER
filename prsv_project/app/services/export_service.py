from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from app.schemas import BatchResult
from app.utils.display_labels import prediction_display_label


class ExportService:
    """
    Creates exportable summary artifacts and charts for batch runs.
    """

    def export_batch_csv(self, run_dir: Path, batch_result: BatchResult) -> Path:
        rows = []
        for item in batch_result.results:
            rows.append(
                {
                    "image_id": item.image_id,
                    "filename": item.filename,
                    "prediction": prediction_display_label(item.prediction),
                    "confidence": item.confidence,
                    "infection_percentage": item.infection_percentage,
                    "severity_score": item.severity_score,
                    "severity_label": item.severity_label,
                }
            )

        csv_path = run_dir / "batch_summary.csv"
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)
        return csv_path

    def create_run_bundle(self, run_dir: Path) -> Path:
        archive_base = run_dir / "export_bundle"
        archive_path = shutil.make_archive(
            base_name=str(archive_base),
            format="zip",
            root_dir=str(run_dir),
        )
        return Path(archive_path)

    def generate_batch_charts(self, run_dir: Path, batch_result: BatchResult) -> Dict[str, str]:
        charts_dir = run_dir / "charts"
        charts_dir.mkdir(parents=True, exist_ok=True)

        prediction_chart_path = charts_dir / "prediction_distribution.png"
        severity_chart_path = charts_dir / "severity_distribution.png"

        healthy_count = batch_result.healthy_count
        diseased_count = batch_result.diseased_count

        fig1, ax1 = plt.subplots(figsize=(6, 4))
        categories = ["Healthy", "PRSV Suspected"]
        values = [healthy_count, diseased_count]
        ax1.bar(categories, values)
        ax1.set_title("Prediction Distribution")
        ax1.set_ylabel("Count")
        for i, value in enumerate(values):
            ax1.text(i, value + 0.05, str(value), ha="center", va="bottom")
        fig1.tight_layout()
        fig1.savefig(prediction_chart_path, dpi=200)
        plt.close(fig1)

        severity_labels = list(batch_result.severity_distribution.keys())
        severity_values = list(batch_result.severity_distribution.values())

        fig2, ax2 = plt.subplots(figsize=(7, 4))
        if severity_labels:
            ax2.bar(severity_labels, severity_values)
            ax2.set_title("Severity Distribution")
            ax2.set_ylabel("Count")
            ax2.tick_params(axis="x", rotation=20)
            for i, value in enumerate(severity_values):
                ax2.text(i, value + 0.05, str(value), ha="center", va="bottom")
        else:
            ax2.text(0.5, 0.5, "No severity data", ha="center", va="center")
            ax2.set_axis_off()

        fig2.tight_layout()
        fig2.savefig(severity_chart_path, dpi=200)
        plt.close(fig2)

        return {
            "prediction_distribution": str(prediction_chart_path),
            "severity_distribution": str(severity_chart_path),
        }
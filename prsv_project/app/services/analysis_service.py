from __future__ import annotations

from pathlib import Path
from typing import Dict

from app.config import Settings
from app.schemas import ImageResult
from app.services.rag_service import RagService
from app.services.run_manager import RunContext, RunManager
from app.utils.file_utils import copy_file
from app.utils.id_utils import generate_image_id
from app.utils.image_utils import read_image_cv, save_image_cv, save_image_pil
from app.utils.json_utils import save_json
from app.utils.logging_utils import get_logger
from image_processing.feature_extraction import extract_handcrafted_features
from image_processing.heatmaps import generate_heatmaps
from image_processing.preprocess import preprocess_image
from image_processing.quality_checks import assess_image_quality
from image_processing.segmentation import segment_leaf
from image_processing.segmentation_quality import assess_segmentation_quality
from image_processing.severity import estimate_severity
from image_processing.symptom_enhancement import enhance_symptoms
from ml.infer_svm import predict_with_svm


class AnalysisService:
    """
    Full end-to-end image analysis orchestrator for PRSV research workflow.
    """

    def __init__(self, settings: Settings, run_manager: RunManager) -> None:
        self.settings = settings
        self.run_manager = run_manager
        self.rag_service = RagService(settings)

    def analyze_single_image(self, image_path: Path, run_context: RunContext | None = None) -> ImageResult:
        if run_context is None:
            run_context = self.run_manager.create_run()

        logger = get_logger("analysis_service", run_context.log_file)

        image_id = generate_image_id()
        image_output_dir = self.run_manager.create_image_dir(run_context, image_id)

        logger.info("Starting analysis for image: %s", image_path)

        original_copy = copy_file(image_path, image_output_dir, filename="original" + image_path.suffix.lower())

        image_bgr = read_image_cv(image_path)
        quality_result = assess_image_quality(image_bgr)

        preprocess_result = preprocess_image(image_bgr, self.settings)
        segmentation_result = segment_leaf(preprocess_result.enhanced_rgb)
        segmentation_quality = assess_segmentation_quality(segmentation_result.mask)

        symptom_result = enhance_symptoms(
            image_rgb=preprocess_result.enhanced_rgb,
            leaf_mask=segmentation_result.mask,
        )

        feature_result = extract_handcrafted_features(
            image_rgb=preprocess_result.enhanced_rgb,
            grayscale=preprocess_result.grayscale,
            hsv=preprocess_result.hsv,
            edge_map=symptom_result.edge_map,
            mask=segmentation_result.mask,
        )

        severity_result = estimate_severity(
            feature_dict=feature_result.feature_dict,
            symptom_mask=symptom_result.symptom_mask,
            leaf_mask=segmentation_result.mask,
            settings=self.settings,
        )

        inference_result = predict_with_svm(
            feature_vector=feature_result.feature_vector,
            settings=self.settings,
        )

        heatmap_result = generate_heatmaps(
            image_rgb=preprocess_result.enhanced_rgb,
            edge_map=symptom_result.edge_map,
            gradient_magnitude=symptom_result.gradient_magnitude,
            laplacian_response=symptom_result.laplacian_response,
            symptom_mask=symptom_result.symptom_mask,
            leaf_mask=segmentation_result.mask,
        )

        symptom_findings: Dict[str, float] = {
            "abnormal_color_score": symptom_result.abnormal_color_score,
            "symptom_region_ratio": severity_result.symptom_region_ratio,
        }

        explanation_trace = self.rag_service.build_explanation_trace(
            prediction=inference_result.prediction,
            confidence=inference_result.confidence,
            severity_label=severity_result.severity_label,
            severity_score=severity_result.severity_score,
            feature_values=feature_result.feature_dict,
            symptom_findings=symptom_findings,
            segmentation_success=segmentation_result.success,
        )

        preprocessed_path = image_output_dir / "preprocessed.jpg"
        segmented_path = image_output_dir / "segmented.jpg"
        contour_path = image_output_dir / "segmentation_contour.jpg"
        symptom_highlight_path = image_output_dir / "symptom_highlighted.jpg"
        edge_heatmap_path = image_output_dir / "edge_heatmap.jpg"
        edge_overlay_path = image_output_dir / "edge_overlay.jpg"
        severity_heatmap_path = image_output_dir / "severity_heatmap.jpg"
        severity_overlay_path = image_output_dir / "severity_overlay.jpg"
        mask_path = image_output_dir / "leaf_mask.png"
        features_json_path = image_output_dir / "features.json"
        prediction_json_path = image_output_dir / "prediction.json"
        severity_json_path = image_output_dir / "severity.json"
        rag_json_path = image_output_dir / "rag.json"
        explanation_json_path = image_output_dir / "explanation.json"

        save_image_pil(preprocessed_path, preprocess_result.enhanced_rgb)
        save_image_pil(segmented_path, segmentation_result.segmented_rgb)
        save_image_pil(contour_path, segmentation_result.contour_visualization_rgb)
        save_image_pil(symptom_highlight_path, symptom_result.symptom_highlight_rgb)
        save_image_cv(edge_heatmap_path, heatmap_result.edge_heatmap_bgr)
        save_image_pil(edge_overlay_path, heatmap_result.edge_overlay_rgb)
        save_image_cv(severity_heatmap_path, heatmap_result.severity_heatmap_bgr)
        save_image_pil(severity_overlay_path, heatmap_result.severity_overlay_rgb)
        save_image_cv(mask_path, segmentation_result.mask)

        features_payload = {
            "image_id": image_id,
            "filename": image_path.name,
            "feature_vector": feature_result.feature_vector,
            "feature_values": feature_result.feature_dict,
            "diagnostics": feature_result.diagnostics,
            "preprocess_metadata": preprocess_result.metadata,
            "quality_assessment": {
                "quality_status": quality_result.quality_status,
                "warnings": quality_result.warnings,
                "metrics": quality_result.metrics,
                "is_small_image": quality_result.is_small_image,
            },
            "segmentation": {
                "leaf_area_ratio": segmentation_result.leaf_area_ratio,
                "contour_count": segmentation_result.contour_count,
                "used_fallback": segmentation_result.used_fallback,
                "success": segmentation_result.success,
                "quality_status": segmentation_quality.quality_status,
                "quality_score": segmentation_quality.quality_score,
                "quality_warnings": segmentation_quality.warnings,
                "quality_metrics": segmentation_quality.metrics,
            },
            "symptom_analysis": symptom_findings,
        }

        prediction_payload = {
            "image_id": image_id,
            "filename": image_path.name,
            "prediction": inference_result.prediction,
            "confidence": inference_result.confidence,
            "probabilities": inference_result.probabilities,
            "inference_mode": inference_result.inference_mode,
            "feature_vector_used": inference_result.feature_vector_used,
        }

        severity_payload = {
            "image_id": image_id,
            "infection_percentage": severity_result.infection_percentage,
            "severity_score": severity_result.severity_score,
            "severity_label": severity_result.severity_label,
            "severity_confidence": severity_result.severity_confidence,
            "symptom_region_ratio": severity_result.symptom_region_ratio,
            "reasoning_trace": severity_result.reasoning_trace,
        }

        rag_payload = {
            "observation_query": explanation_trace.observation_query,
            "retrieved_evidence": [item.model_dump() for item in explanation_trace.retrieved_evidence],
            "technical_explanation": explanation_trace.technical_explanation,
            "farmer_friendly_explanation": explanation_trace.farmer_friendly_explanation,
            "advisory_notes": explanation_trace.advisory_notes,
            "status": "active_local_rag",
        }

        save_json(features_json_path, features_payload)
        save_json(prediction_json_path, prediction_payload)
        save_json(severity_json_path, severity_payload)
        save_json(rag_json_path, rag_payload)
        save_json(explanation_json_path, explanation_trace.model_dump())

        output_paths = {
            "original": str(original_copy),
            "preprocessed": str(preprocessed_path),
            "segmented": str(segmented_path),
            "segmentation_contour": str(contour_path),
            "symptom_highlighted": str(symptom_highlight_path),
            "leaf_mask": str(mask_path),
            "edge_heatmap": str(edge_heatmap_path),
            "edge_overlay": str(edge_overlay_path),
            "severity_heatmap": str(severity_heatmap_path),
            "severity_overlay": str(severity_overlay_path),
            "features_json": str(features_json_path),
            "prediction_json": str(prediction_json_path),
            "severity_json": str(severity_json_path),
            "rag_json": str(rag_json_path),
            "explanation_json": str(explanation_json_path),
        }

        logger.info(
            "Analysis completed for image=%s | prediction=%s | confidence=%.4f | severity=%s | quality=%s | segmentation=%s",
            image_path.name,
            inference_result.prediction,
            inference_result.confidence,
            severity_result.severity_label,
            quality_result.quality_status,
            segmentation_quality.quality_status,
        )

        return ImageResult(
            image_id=image_id,
            filename=image_path.name,
            prediction=inference_result.prediction,
            confidence=inference_result.confidence,
            infection_percentage=severity_result.infection_percentage,
            severity_score=severity_result.severity_score,
            severity_label=severity_result.severity_label,
            feature_values=feature_result.feature_dict,
            output_paths=output_paths,
            explanation_trace=explanation_trace,
        )
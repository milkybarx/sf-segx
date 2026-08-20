"""Unified Mask2Former Phase 2 scientific analysis pipeline."""
from pathlib import Path
from typing import Dict, Optional
import cv2
import numpy as np

from catalog.exporter import export_catalog
from catalog.schema import build_catalog
from catalog.validator import validate_catalog
from postprocessing.calibration import physical_measurements
from postprocessing.instances import separate_filaments
from postprocessing.skeleton import analyze_skeleton
from postprocessing.spatial import add_spatial_metadata
from postprocessing.thresholding import probability_to_mask
from inference.mask2former import Mask2FormerInference, run_mask2former_inference
from inference.adapters import StandardizedPrediction, get_segmentation_adapter


def morphology_risk_screening(filament: Dict) -> str:
    """Return a morphology-based screening band, not an eruption or CME probability."""
    score = (min(filament.get("skeleton_length_px", 0) / 500.0, 1.0) * 0.45
             + min(filament.get("area_px", 0) / 10000.0, 1.0) * 0.30
             + min(filament.get("confidence", 0), 1.0) * 0.25)
    return "HIGH" if score >= 0.75 else "MODERATE" if score >= 0.45 else "LOW"


def run_phase2_analysis(image: np.ndarray, image_id: str = "image",
                        model=None, model_name: str = "mask2former", threshold: float = 0.5, min_area: int = 50,
                        calibration: Optional[Dict] = None, timestamp: Optional[str] = None,
                        explain: bool = False, output_dir: Optional[str | Path] = None) -> Dict:
    """Run common Phase 2 analysis on the selected model's standardized prediction."""
    adapter = get_segmentation_adapter(model_name, model)
    if model is None and explain and adapter.supports_explainability:
        from model_hub import get_model
        model, _ = get_model(model_name)
        adapter.model = model
    prediction: StandardizedPrediction = adapter.predict(image, threshold)
    mask = probability_to_mask(prediction.probability, threshold)
    labels, filaments = separate_filaments(mask, prediction.probability, min_area)
    attribution = None
    if explain:
        from explainability.interface import generate_explanation
        attribution = generate_explanation(model, image, prediction, model_name) if adapter.supports_explainability and model is not None else None
    for filament in filaments:
        filament.update(analyze_skeleton(filament.pop("component_mask")))
        add_spatial_metadata(filament, image.shape[:2])
        filament["image_width"] = int(image.shape[1])
        filament["image_height"] = int(image.shape[0])
        filament["physical"] = physical_measurements(filament["skeleton_length_px"], filament["area_px"], calibration)
        filament["risk_screening_indicator"] = morphology_risk_screening(filament)
    records = build_catalog(filaments, image_id, timestamp, model_name=model_name,
                             model_checkpoint=prediction.model_checkpoint, threshold=threshold)
    validate_catalog(records)
    paths = None
    figure = None
    if output_dir is not None:
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        from visualization.phase2 import create_phase2_figure
        figure = create_phase2_figure(image, prediction.probability, mask, filaments, attribution,
                                      directory / "phase2_analysis.png")
        figure.clf()
        from visualization.phase2 import save_filament_crops
        crop_paths = save_filament_crops(image, filaments, directory / "filament_crops")
        paths = export_catalog(records, directory / "catalog")
    else:
        crop_paths = []
    return {"image_id": image_id, "model": model_name, "model_name": model_name,
            "model_checkpoint": prediction.model_checkpoint, "threshold": threshold,
            "prediction": prediction, "inference": prediction, "labels": labels,
            "filaments": filaments, "catalog": records, "attribution": attribution,
            "explainability_supported": adapter.supports_explainability,
            "exports": paths, "crop_paths": crop_paths}

"""Dependency-light sanity tests for Phase 2 scientific post-processing."""
import json
import numpy as np
from catalog.exporter import export_catalog
from catalog.schema import filament_record
from catalog.validator import validate_catalog
from postprocessing.calibration import physical_measurements
from postprocessing.skeleton import analyze_skeleton
from postprocessing.spatial import spatial_region
from postprocessing.thresholding import probability_to_mask, segmentation_metrics
from visualization.phase2 import _instance_panel
from visualization.detail import crop_filament, super_resolve_crop, selected_overlay
from inference.adapters import StandardizedPrediction


def test_probability_metrics_and_mask():
    probability = np.array([[0.1, 0.8], [0.6, 0.2]], dtype=np.float32)
    mask = probability_to_mask(probability, 0.5)
    assert mask.dtype == np.uint8 and set(np.unique(mask)) == {0, 1}
    assert segmentation_metrics(mask, mask)["dice"] > 0.99


def test_skeleton_degenerate_and_sinuosity():
    line = np.zeros((9, 9), dtype=np.uint8)
    line[4, 1:8] = 1
    result = analyze_skeleton(line)
    assert result["skeleton_length_px"] > 0
    assert result["sinuosity"] >= 1.0


def test_spatial_and_missing_calibration():
    assert spatial_region({"x": 5, "y": 5}, 30, 30) == "NW"
    physical = physical_measurements(10, 20)
    assert physical == {"calibrated": False, "length_km": None, "area_km2": None}


def test_catalog_export(tmp_path):
    record = filament_record({"filament_id": 1, "confidence": 0.9, "area_px": 10,
                              "perimeter_px": 12, "skeleton_length_px": 8, "avg_width_px": 1.2,
                              "sinuosity": 1.1, "orientation_deg": 20, "centroid": {"x": 2, "y": 3},
                              "bbox": {"x": 1, "y": 1, "width": 2, "height": 3}, "spatial_region": "NW"}, "sample")
    validate_catalog([record])
    json_path, csv_path = export_catalog([record], tmp_path)
    assert json.loads(json_path.read_text())[0]["model"] == "mask2former"
    assert csv_path.exists()


def test_visualization_resizes_original_resolution_skeleton():
    """Dashboard-sized panels must accept morphology masks from the source image."""
    image = np.zeros((512, 512), dtype=np.uint8)
    skeleton = np.zeros((2048, 2048), dtype=np.uint8)
    skeleton[100:110, 100:110] = 1
    filament = {"filament_id": 1, "spatial_region": "NW",
                "image_width": 2048, "image_height": 2048,
                "bbox": {"x_min": 100, "y_min": 100, "x_max": 200, "y_max": 200},
                "skeleton_mask": skeleton}
    panel = _instance_panel(image, [filament], skeleton=True)
    assert panel.shape == (512, 512, 3)


def test_detail_crop_and_optional_enhancement():
    image = np.zeros((100, 140), dtype=np.uint8)
    labels = np.zeros_like(image, dtype=np.int32)
    labels[30:60, 50:90] = 4
    filament = {
        "filament_id": 1, "component_id": 4, "confidence": 0.9,
        "spatial_region": "CENTER", "image_width": 140, "image_height": 100,
        "bbox": {"x_min": 50, "y_min": 30, "x_max": 90, "y_max": 60},
        "skeleton_mask": labels.astype(np.uint8),
    }
    crop, bounds = crop_filament(image, filament, padding=20)
    assert crop.shape == (70, 80)
    enhanced = super_resolve_crop(crop, scale=2)
    assert enhanced.shape == (140, 160)
    overlay = selected_overlay(crop, filament, labels, bounds,
                               show_mask=False, show_skeleton=False,
                               show_bbox=False, show_labels=False)
    assert overlay.shape == (70, 80, 3)


def test_standardized_prediction_contract():
    mask = np.zeros((16, 20), dtype=np.uint8)
    probability = np.zeros((16, 20), dtype=np.float32)
    prediction = StandardizedPrediction(
        mask=mask, probability=probability, confidence=0.0,
        model_name="segformer_b0", model_checkpoint="checkpoint.pt",
        image=np.zeros((16, 20), dtype=np.uint8), preprocessed=np.zeros((8, 8), dtype=np.uint8),
        disk_mask=np.ones((16, 20), dtype=bool),
    )
    assert prediction.mask.shape == prediction.probability.shape
    assert prediction.model_name == "segformer_b0"

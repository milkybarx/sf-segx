"""Serializable catalog records for Phase 2."""
from typing import Dict, List, Optional


def filament_record(filament: Dict, image_id: str, timestamp: Optional[str] = None,
                    physical: Optional[Dict] = None) -> Dict:
    """Create a stable JSON/CSV-ready filament record."""
    bbox = filament.get("bbox", {})
    record = {"image_id": image_id, "timestamp": timestamp, "model": "mask2former",
              "filament_id": int(filament["filament_id"]), "confidence": float(filament.get("confidence", 0.0)),
              "area_px": float(filament.get("area_px", 0.0)), "perimeter_px": float(filament.get("perimeter_px", 0.0)),
              "skeleton_length_px": float(filament.get("skeleton_length_px", 0.0)),
              "average_width_px": float(filament.get("avg_width_px", 0.0)),
              "sinuosity": float(filament.get("sinuosity", 1.0)),
              "orientation_deg": float(filament.get("orientation_deg", 0.0)),
              "centroid": filament.get("centroid", {}),
              "bbox": {"x_min": int(bbox.get("x_min", bbox.get("x", 0))), "y_min": int(bbox.get("y_min", bbox.get("y", 0))),
                       "x_max": int(bbox.get("x_max", bbox.get("x", 0) + bbox.get("width", 0))),
                       "y_max": int(bbox.get("y_max", bbox.get("y", 0) + bbox.get("height", 0))),
                       "width": int(bbox.get("width", 0)), "height": int(bbox.get("height", 0))},
              "spatial_region": filament.get("spatial_region", "CENTER"),
              "physical": physical or {"calibrated": False, "length_km": None, "area_km2": None}}
    return record


def build_catalog(filaments: List[Dict], image_id: str, timestamp: Optional[str] = None) -> List[Dict]:
    """Build records while excluding internal NumPy component masks."""
    return [filament_record(f, image_id, timestamp, f.get("physical")) for f in filaments]

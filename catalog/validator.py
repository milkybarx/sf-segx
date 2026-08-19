"""Lightweight catalog schema validation."""
from typing import Dict, Iterable

REQUIRED = {"image_id", "model", "filament_id", "confidence", "area_px", "perimeter_px",
            "skeleton_length_px", "average_width_px", "sinuosity", "orientation_deg", "centroid",
            "bbox", "spatial_region", "physical"}


def validate_record(record: Dict) -> None:
    """Raise ValueError when a catalog record is incomplete or physically invalid."""
    missing = REQUIRED.difference(record)
    if missing:
        raise ValueError(f"catalog record missing fields: {sorted(missing)}")
    if record["model"] != "mask2former":
        raise ValueError("Phase 2 catalog model must be mask2former")
    physical = record["physical"]
    if not physical.get("calibrated") and (physical.get("length_km") is not None or physical.get("area_km2") is not None):
        raise ValueError("uncalibrated records cannot contain physical measurements")


def validate_catalog(records: Iterable[Dict]) -> bool:
    """Validate all records and return True."""
    for record in records:
        validate_record(record)
    return True

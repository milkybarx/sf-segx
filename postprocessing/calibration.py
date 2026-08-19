"""Explicit, metadata-gated pixel to physical-unit calibration."""
from typing import Dict, Optional


def calibration_from_metadata(metadata: Optional[Dict]) -> Dict:
    """Return calibration status; never infer a solar scale without metadata."""
    metadata = metadata or {}
    km_per_px = metadata.get("km_per_px")
    if km_per_px is None and metadata.get("solar_radius_px") and metadata.get("solar_radius_km"):
        km_per_px = float(metadata["solar_radius_km"]) / float(metadata["solar_radius_px"])
    if km_per_px is None or float(km_per_px) <= 0:
        return {"calibrated": False, "km_per_px": None}
    return {"calibrated": True, "km_per_px": float(km_per_px)}


def physical_measurements(length_px: float, area_px: float,
                          metadata: Optional[Dict] = None) -> Dict:
    """Convert length and area only when calibration is valid."""
    calibration = calibration_from_metadata(metadata)
    scale = calibration["km_per_px"]
    return {"calibrated": calibration["calibrated"],
            "length_km": length_px * scale if scale else None,
            "area_km2": area_px * scale ** 2 if scale else None}

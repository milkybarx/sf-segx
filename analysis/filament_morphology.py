"""
Filament Morphology Analysis
=============================
Comprehensive geometric analysis of detected solar filaments.
All measurements are in PIXEL units (no physical calibration).
"""

import numpy as np
import cv2
from typing import List, Dict
from classical.morphology import (
    connected_components_analysis,
    extract_skeleton,
    measure_filament_properties
)


def analyze_filaments(mask: np.ndarray, probability_map: np.ndarray = None,
                       min_area: int = 50) -> List[Dict]:
    """
    Analyze all detected filaments in a segmentation mask.

    Args:
        mask: Binary segmentation mask
        probability_map: Probability map for confidence estimation
        min_area: Minimum area to consider as a filament

    Returns:
        List of filament property dicts
    """
    labels, components = connected_components_analysis(mask, min_area=min_area)

    filaments = []
    for comp in components:
        comp_id = comp['id']
        component_mask = (labels == comp_id).astype(np.uint8)

        props = measure_filament_properties(mask, component_mask, probability_map)
        if props:
            props['filament_id'] = len(filaments) + 1
            props['component_id'] = comp_id
            filaments.append(props)

    # Sort by area (largest first)
    filaments.sort(key=lambda x: x.get('area_px', 0), reverse=True)

    return filaments


def generate_morphology_report(filaments: List[Dict]) -> str:
    """Generate a text summary of filament morphology analysis."""
    if not filaments:
        return "No filaments detected."

    lines = []
    lines.append(f"{'='*60}")
    lines.append(f"SOLAR FILAMENT MORPHOLOGY REPORT")
    lines.append(f"{'='*60}")
    lines.append(f"Total filaments detected: {len(filaments)}")
    lines.append(f"")

    total_area = sum(f.get('area_px', 0) for f in filaments)
    lines.append(f"Total filament area: {total_area:.0f} pixels")
    lines.append(f"")

    for f in filaments:
        lines.append(f"--- Filament #{f['filament_id']} ---")
        lines.append(f"  Area:           {f.get('area_px', 0):.0f} px")
        lines.append(f"  Perimeter:      {f.get('perimeter_px', 0):.0f} px")
        lines.append(f"  Length (skel):   {f.get('skeleton_length_px', 0):.0f} px")
        lines.append(f"  Avg Width:       {f.get('avg_width_px', 0):.1f} px")
        lines.append(f"  Orientation:     {f.get('orientation_deg', 0):.1f}°")
        bbox = f.get('bbox', {})
        lines.append(f"  Bounding Box:    ({bbox.get('x',0)}, {bbox.get('y',0)}) "
                     f"{bbox.get('width',0)}×{bbox.get('height',0)}")
        cent = f.get('centroid', {})
        lines.append(f"  Centroid:        ({cent.get('x',0):.1f}, {cent.get('y',0):.1f})")
        lines.append(f"  Confidence:      {f.get('confidence', 0):.3f}")
        lines.append(f"  Unit:            {f.get('unit', 'pixels')}")
        lines.append(f"")

    return "\n".join(lines)


def draw_morphology_overlay(image: np.ndarray, mask: np.ndarray,
                              filaments: List[Dict]) -> np.ndarray:
    """
    Draw morphology annotations on the image.
    Shows bounding boxes, centroids, orientation, and IDs.
    """
    if len(image.shape) == 2:
        vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        vis = image.copy()

    for f in filaments:
        bbox = f.get('bbox', {})
        x, y, w, h = bbox.get('x', 0), bbox.get('y', 0), bbox.get('width', 0), bbox.get('height', 0)
        cent = f.get('centroid', {})
        cx, cy = int(cent.get('x', 0)), int(cent.get('y', 0))

        # Bounding box
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 1)

        # Centroid
        cv2.circle(vis, (cx, cy), 3, (0, 0, 255), -1)

        # ID label
        cv2.putText(vis, f"#{f['filament_id']}", (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # Orientation line
        angle_rad = np.deg2rad(f.get('orientation_deg', 0))
        line_len = max(w, h) // 2
        dx = int(line_len * np.cos(angle_rad))
        dy = int(line_len * np.sin(angle_rad))
        cv2.line(vis, (cx - dx, cy - dy), (cx + dx, cy + dy), (255, 255, 0), 1)

    return vis


def export_morphology_csv(filaments: List[Dict], output_path: str):
    """Export filament morphology metrics to a CSV spreadsheet."""
    import csv
    with open(output_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Filament_ID", "Area_px", "Area_km2", "Perimeter_px",
            "Length_skel_px", "Length_km", "Avg_Width_px",
            "Orientation_deg", "Centroid_X", "Centroid_Y",
            "BBox_X", "BBox_Y", "BBox_W", "BBox_H", "Confidence"
        ])
        for comp in filaments:
            bbox = comp.get('bbox', {})
            cent = comp.get('centroid', {})
            area_px = comp.get('area_px', 0)
            length_px = comp.get('skeleton_length_px', 0)

            # Astronomical scaling: 435.0 km / pixel
            km_per_px = comp.get('km_per_px', 435.0)
            area_km2 = area_px * (km_per_px ** 2)
            length_km = length_px * km_per_px

            writer.writerow([
                comp.get('filament_id', 0),
                round(area_px, 2),
                round(area_km2, 2),
                round(comp.get('perimeter_px', 0), 2),
                round(length_px, 2),
                round(length_km, 2),
                round(comp.get('avg_width_px', 0), 2),
                round(comp.get('orientation_deg', 0), 2),
                round(cent.get('x', 0), 2),
                round(cent.get('y', 0), 2),
                bbox.get('x', 0),
                bbox.get('y', 0),
                bbox.get('width', 0),
                bbox.get('height', 0),
                round(comp.get('confidence', 0), 3)
            ])


def export_morphology_json(filaments: List[Dict], output_path: str):
    """Export filament morphology metrics to a structured JSON file."""
    import json
    with open(output_path, mode='w', encoding='utf-8') as f:
        json.dump(filaments, f, indent=2)


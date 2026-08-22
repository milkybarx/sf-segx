"""
Filament Morphology Analysis
=============================
Comprehensive geometric analysis of detected solar filaments.
All measurements are in PIXEL units (no physical calibration).
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple
from classical.morphology import (
    connected_components_analysis,
    extract_skeleton,
    measure_filament_properties
)
from analysis.coordinates import pixel_to_stonyhurst


def get_disk_params(disk_mask: np.ndarray) -> Tuple[float, float, float]:
    """Retrieve solar disk center and radius from binary disk mask."""
    ys, xs = np.where(disk_mask)
    if len(xs) == 0:
        h, w = disk_mask.shape[:2]
        return w / 2.0, h / 2.0, min(h, w) / 2.0
    cx = (xs.min() + xs.max()) / 2.0
    cy = (ys.min() + ys.max()) / 2.0
    radius = ((xs.max() - xs.min()) + (ys.max() - ys.min())) / 4.0
    return float(cx), float(cy), float(radius)


def enrich_filament_properties(filament: Dict, probability_map: np.ndarray, disk_mask: np.ndarray) -> Dict:
    """Enrich the filament dictionary with extended morphological, statistical, and heliographic properties."""
    area_px = filament.get("area_px", 0.0)
    perimeter_px = filament.get("perimeter_px", 0.0)
    skeleton_length_px = filament.get("skeleton_length_px", 0.0)
    avg_width_px = filament.get("avg_width_px", 0.0)
    major = filament.get("major_axis_px", 0.0)
    minor = filament.get("minor_axis_px", 0.0)
    
    # 1. Ratios and Compactness
    filament["aspect_ratio"] = float(major / minor) if minor > 0 else float("nan")
    filament["length_width_ratio"] = float(skeleton_length_px / avg_width_px) if avg_width_px > 0 else float("nan")
    filament["perimeter_area_ratio"] = float(perimeter_px / area_px) if area_px > 0 else float("nan")
    filament["skeleton_area_ratio"] = float(skeleton_length_px / area_px) if area_px > 0 else float("nan")
    filament["compactness"] = float(4.0 * np.pi * area_px / (perimeter_px ** 2)) if perimeter_px > 0 else 0.0
    
    # 2. Probability map statistics
    if probability_map is not None:
        # We need the original component mask, or we can use the bbox to crop the prob map
        # Let's crop it to the bounding box or use centroid-based/mask-based calculations
        # Since we have component_mask in downstream separate_filaments, we can pass it,
        # but if we don't have it, we can estimate statistics over the bbox area
        bbox = filament.get("bbox", {})
        x_min, y_min = int(bbox.get("x_min", bbox.get("x", 0))), int(bbox.get("y_min", bbox.get("y", 0)))
        x_max = int(bbox.get("x_max", x_min + bbox.get("width", 0)))
        y_max = int(bbox.get("y_max", y_min + bbox.get("height", 0)))
        
        # Clip bbox to probability map dimensions
        h_prob, w_prob = probability_map.shape[:2]
        x1, y1 = max(0, x_min), max(0, y_min)
        x2, y2 = min(w_prob, x_max), min(h_prob, y_max)
        
        if x2 > x1 and y2 > y1:
            crop = probability_map[y1:y2, x1:x2]
            filament["prob_min"] = float(np.min(crop))
            filament["prob_max"] = float(np.max(crop))
            filament["prob_std"] = float(np.std(crop))
            filament["prob_median"] = float(np.median(crop))
            filament["prob_p90"] = float(np.percentile(crop, 90))
        else:
            filament["prob_min"] = float("nan")
            filament["prob_max"] = float("nan")
            filament["prob_std"] = float("nan")
            filament["prob_median"] = float("nan")
            filament["prob_p90"] = float("nan")
    else:
        for k in ["prob_min", "prob_max", "prob_std", "prob_median", "prob_p90"]:
            filament[k] = float("nan")

    # 3. Disk Position and Heliographic Coordinates
    cx_disk, cy_disk, r_disk = get_disk_params(disk_mask)
    centroid = filament.get("centroid", {})
    cx_fil = centroid.get("x", cx_disk)
    cy_fil = centroid.get("y", cy_disk)
    
    # Fractional distance from disk center
    dist_px = np.sqrt((cx_fil - cx_disk)**2 + (cy_fil - cy_disk)**2)
    filament["disk_center_dist"] = float(dist_px / r_disk) if r_disk > 0 else float("nan")
    
    # Stonyhurst Latitude and Longitude
    lat, lon = pixel_to_stonyhurst(cx_fil, cy_fil, cx_disk, cy_disk, r_disk)
    filament["solar_coordinates"] = {"latitude": lat, "longitude": lon}
    filament["solar_lat"] = lat
    filament["solar_lon"] = lon
    
    # 4. Optional placeholder/historical fields (initialized to None/NaN)
    filament["active_region"] = None
    filament["eruption_indicator"] = None
    filament["filament_type"] = None
    filament["rating"] = None
    filament["orientation_stability"] = float("nan")
    
    return filament



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
            props['component_id'] = comp_id
            props['filament_id'] = len(filaments) + 1
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


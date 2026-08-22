"""
Morphology Analysis
===================
Connected component analysis, skeleton extraction, and measurement
utilities for solar filament detection and analysis.
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple


def connected_components_analysis(mask: np.ndarray, min_area: int = 50) -> Tuple[np.ndarray, List[Dict]]:
    """
    Perform connected component analysis on a binary mask.

    Returns:
        labeled: Labeled image where each filament has a unique ID
        components: List of dicts with properties for each component
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )

    components = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue

        component = {
            'id': i,
            'area': int(area),
            'bbox': {
                'x': int(stats[i, cv2.CC_STAT_LEFT]),
                'y': int(stats[i, cv2.CC_STAT_TOP]),
                'width': int(stats[i, cv2.CC_STAT_WIDTH]),
                'height': int(stats[i, cv2.CC_STAT_HEIGHT]),
            },
            'centroid': {
                'x': float(centroids[i][0]),
                'y': float(centroids[i][1]),
            },
        }
        components.append(component)

    return labels, components


def extract_skeleton(mask: np.ndarray) -> np.ndarray:
    """
    Extract the morphological skeleton of a binary mask.
    Uses iterative thinning.
    """
    skel = np.zeros_like(mask, dtype=np.uint8)
    temp = mask.copy().astype(np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    while True:
        eroded = cv2.erode(temp, element)
        opened = cv2.dilate(eroded, element)
        subset = temp - opened
        skel = cv2.bitwise_or(skel, subset)
        temp = eroded.copy()
        if cv2.countNonZero(temp) == 0:
            break

    return skel


def measure_filament_properties(mask: np.ndarray, component_mask: np.ndarray,
                                 probability_map: np.ndarray = None) -> Dict:
    """
    Measure geometric properties of a single filament region.
    All measurements are in PIXEL units.
    """
    # Find contours for this component
    contours, _ = cv2.findContours(
        component_mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return {}

    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)

    # Bounding box
    x, y, w, h = cv2.boundingRect(contour)

    # Moments for centroid
    M = cv2.moments(contour)
    cx = M['m10'] / (M['m00'] + 1e-10)
    cy = M['m01'] / (M['m00'] + 1e-10)

    # Fit ellipse if enough points
    orientation = 0.0
    major_axis = 0.0
    minor_axis = 0.0
    if len(contour) >= 5:
        ellipse = cv2.fitEllipse(contour)
        (ecx, ecy), (ma, MA), angle = ellipse
        orientation = angle
        major_axis = max(ma, MA)
        minor_axis = min(ma, MA)

    # Skeleton for length estimation
    skeleton = extract_skeleton(component_mask)
    skeleton_length = float(np.sum(skeleton > 0))

    # Average width (area / skeleton_length)
    avg_width = area / max(skeleton_length, 1)

    # Perimeter
    perimeter = cv2.arcLength(contour, closed=True)

    # Confidence (mean probability if available)
    confidence = 0.0
    if probability_map is not None:
        prob_vals = probability_map[component_mask > 0]
        if len(prob_vals) > 0:
            confidence = float(np.mean(prob_vals))

    return {
        'area_px': float(area),
        'perimeter_px': float(perimeter),
        'bbox': {'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h)},
        'centroid': {'x': float(cx), 'y': float(cy)},
        'orientation_deg': float(orientation),
        'major_axis_px': float(major_axis),
        'minor_axis_px': float(minor_axis),
        'skeleton_length_px': float(skeleton_length),
        'avg_width_px': float(avg_width),
        'confidence': float(confidence),
        'unit': 'pixels (no physical calibration)',
    }

"""
Explainability Module
=====================
Confidence maps, uncertainty visualization, and high-confidence highlighting.
"""

import numpy as np
import cv2
from typing import Tuple


def compute_confidence_map(probability_map: np.ndarray) -> np.ndarray:
    """
    Compute pixel-level confidence from probability map.
    Confidence = max(p, 1-p) — high near 0 or 1, low near 0.5.
    """
    return np.maximum(probability_map, 1 - probability_map)


def compute_uncertainty_map(probability_map: np.ndarray) -> np.ndarray:
    """
    Compute pixel-level uncertainty (entropy-based).
    High uncertainty near 0.5 (model is unsure).
    """
    p = np.clip(probability_map, 1e-7, 1 - 1e-7)
    entropy = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    return entropy


def high_confidence_mask(probability_map: np.ndarray, threshold: float = 0.8) -> np.ndarray:
    """
    Extract only high-confidence filament detections.
    """
    return (probability_map > threshold).astype(np.uint8)


def create_confidence_overlay(image: np.ndarray, probability_map: np.ndarray,
                                low_threshold: float = 0.3,
                                high_threshold: float = 0.7) -> np.ndarray:
    """
    Create color-coded confidence overlay:
    - Red: High confidence filament (p > high_threshold)
    - Yellow: Medium confidence (low_threshold < p < high_threshold)
    - Green boundary: Low confidence detection
    """
    if len(image.shape) == 2:
        vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        vis = image.copy()

    # Resize prob map if needed
    if probability_map.shape[:2] != vis.shape[:2]:
        probability_map = cv2.resize(probability_map, (vis.shape[1], vis.shape[0]))

    overlay = vis.copy()

    # High confidence (red)
    high = probability_map > high_threshold
    overlay[high] = [0, 0, 255]

    # Medium confidence (yellow)
    medium = (probability_map > low_threshold) & (probability_map <= high_threshold)
    overlay[medium] = [0, 255, 255]

    result = cv2.addWeighted(vis, 0.6, overlay, 0.4, 0)

    return result


def generate_explainability_panel(image: np.ndarray, probability_map: np.ndarray,
                                    target_size: int = 512) -> np.ndarray:
    """
    Generate a 2x2 explainability panel:
    1. Probability heatmap
    2. Confidence map
    3. Uncertainty map
    4. High-confidence overlay
    """
    from visualization.viz import probability_to_heatmap

    # Resize inputs
    img = cv2.resize(image, (target_size, target_size))
    prob = cv2.resize(probability_map, (target_size, target_size))

    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Panel 1: Probability heatmap
    p1 = probability_to_heatmap(prob)
    cv2.putText(p1, "Probability", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Panel 2: Confidence map
    conf = compute_confidence_map(prob)
    p2 = probability_to_heatmap(conf, cv2.COLORMAP_VIRIDIS)
    cv2.putText(p2, "Confidence", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Panel 3: Uncertainty map
    unc = compute_uncertainty_map(prob)
    unc_norm = unc / max(unc.max(), 1e-7)
    p3 = probability_to_heatmap(unc_norm, cv2.COLORMAP_INFERNO)
    cv2.putText(p3, "Uncertainty", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Panel 4: High-confidence overlay
    p4 = create_confidence_overlay(img, prob)
    cv2.putText(p4, "Confidence Overlay", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 2x2 grid
    row1 = np.hstack([p1, p2])
    row2 = np.hstack([p3, p4])
    panel = np.vstack([row1, row2])

    return panel

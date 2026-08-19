"""Semantic-mask to downstream filament instance separation."""
from typing import Dict, List, Tuple
import numpy as np
from analysis.filament_morphology import analyze_filaments
from classical.morphology import connected_components_analysis


def separate_filaments(mask: np.ndarray, probability: np.ndarray,
                       min_area: int = 50) -> Tuple[np.ndarray, List[Dict]]:
    """Label connected components and enrich measurements from existing morphology code.

    The repository checkpoint emits a dense semantic mask; these are downstream components,
    not native Mask2Former panoptic instances.
    """
    labels, components = connected_components_analysis(mask, min_area=min_area)
    filaments = analyze_filaments(mask, probability, min_area=min_area)
    for index, filament in enumerate(filaments, 1):
        filament["filament_id"] = index
        filament["component_mask"] = (labels == filament["component_id"]).astype(np.uint8)
    return labels, filaments

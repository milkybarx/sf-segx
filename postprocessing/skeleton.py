"""Skeleton measurements using the repository's thinning implementation."""
from typing import Dict, List, Tuple
import cv2
import numpy as np
from classical.morphology import extract_skeleton


def _pixels(skeleton: np.ndarray) -> List[Tuple[int, int]]:
    ys, xs = np.where(skeleton > 0)
    return list(zip(xs.tolist(), ys.tolist()))


def analyze_skeleton(component_mask: np.ndarray) -> Dict:
    """Return one-pixel skeleton, endpoints, branches, orientation, and sinuosity."""
    skeleton = extract_skeleton(component_mask)
    points = _pixels(skeleton)
    neighbors = []
    for x, y in points:
        count = int(np.count_nonzero(skeleton[max(0, y-1):y+2, max(0, x-1):x+2])) - 1
        neighbors.append((x, y, count))
    endpoints = [(x, y) for x, y, count in neighbors if count == 1]
    branches = [(x, y) for x, y, count in neighbors if count > 2]
    if len(points) < 2:
        return {"skeleton_mask": skeleton, "skeleton_length_px": float(len(points)),
                "endpoints": endpoints, "branch_points": branches,
                "orientation_deg": 0.0, "sinuosity": 1.0}
    coords = np.asarray(points, dtype=float)
    _, _, vt = np.linalg.svd(coords - coords.mean(axis=0), full_matrices=False)
    orientation = float(np.degrees(np.arctan2(vt[0, 1], vt[0, 0])) % 180)
    if len(endpoints) >= 2:
        distances = [((a[0]-b[0]) ** 2 + (a[1]-b[1]) ** 2) ** 0.5
                     for i, a in enumerate(endpoints) for b in endpoints[i+1:]]
        straight = max(distances, default=0.0)
        sinuosity = float(len(points) / straight) if straight > 0 else 1.0
    else:
        sinuosity = 1.0
    return {"skeleton_mask": skeleton, "skeleton_length_px": float(len(points)),
            "endpoints": endpoints, "branch_points": branches,
            "orientation_deg": orientation, "sinuosity": sinuosity}

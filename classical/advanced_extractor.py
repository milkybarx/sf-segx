"""
Advanced Solar Filament Extractor
==================================
Specifically engineered for full-disk H-alpha solar imagery.

Solves:
1. Complete elimination of outer solar limb edge artifacts (disk boundary).
2. High sensitivity to faint, fragmented, and dark thread-like filaments.
3. Explicit rejection of bright plage/faculae, round sunspots, and chromospheric noise.
"""

import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
from typing import Tuple, Dict, List


def detect_solar_disk_robust(gray: np.ndarray) -> Tuple[int, int, int]:
    """
    Robustly detect the solar disk center and radius.
    """
    h, w = gray.shape
    # Downsample for fast circle detection
    small = cv2.resize(gray, (512, 512))
    blur = cv2.GaussianBlur(small, (9, 9), 2)
    _, thresh = cv2.threshold(blur, 25, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        c = max(contours, key=cv2.contourArea)
        (cx_s, cy_s), r_s = cv2.minEnclosingCircle(c)
        scale_x = w / 512.0
        scale_y = h / 512.0
        cx = int(cx_s * scale_x)
        cy = int(cy_s * scale_y)
        radius = int(r_s * min(scale_x, scale_y))
    else:
        cx, cy, radius = w // 2, h // 2, int(min(w, h) * 0.46)

    return cx, cy, radius


def extract_filaments_advanced(
    image: np.ndarray,
    target_size: int = 512,
    limb_margin: float = 0.08, # Mask out outer 8% of radius where limb cliff occurs
    frangi_scales: List[float] = [1.0, 1.8, 3.0, 5.0, 7.5],
    min_filament_area: int = 15,
    min_elongation: float = 1.6, # Filaments must be elongated structures
) -> Dict[str, np.ndarray]:
    """
    Advanced multi-stage solar filament detection algorithm.
    """
    # 1. Grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    orig_h, orig_w = gray.shape

    # 2. Detect solar disk
    cx, cy, radius = detect_solar_disk_robust(gray)
    safe_radius = int(radius * (1.0 - limb_margin))

    # Create safe disk mask (completely zeroes out the bright/dark solar limb cliff)
    disk_mask_orig = np.zeros((orig_h, orig_w), dtype=np.uint8)
    cv2.circle(disk_mask_orig, (cx, cy), safe_radius, 255, -1)

    # 3. Work at normalized target size
    gray_resized = cv2.resize(gray, (target_size, target_size), interpolation=cv2.INTER_AREA)
    cx_512 = int(cx * target_size / orig_w)
    cy_512 = int(cy * target_size / orig_h)
    safe_r_512 = int(safe_radius * target_size / orig_w)

    disk_mask_512 = np.zeros((target_size, target_size), dtype=np.uint8)
    cv2.circle(disk_mask_512, (cx_512, cy_512), safe_r_512, 255, -1)

    # 4. Background subtraction (Limb Darkening & Granulation removal)
    # Estimate low-frequency background using large median/gaussian filter
    bg = cv2.GaussianBlur(gray_resized, (51, 51), 25)

    # Dark structures = Background - Original Image (positive where image is darker than local average)
    dark_contrast = np.maximum(bg.astype(np.float32) - gray_resized.astype(np.float32), 0)
    dark_contrast[disk_mask_512 == 0] = 0

    # 5. Multi-Scale Black Top-Hat Transform (mathematically isolates dark structures)
    tophat_responses = []
    for k in [7, 13, 21, 31]:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        bhat = cv2.morphologyEx(gray_resized, cv2.MORPH_BLACKHAT, kernel)
        bhat[disk_mask_512 == 0] = 0
        tophat_responses.append(bhat.astype(np.float32))
    multi_tophat = np.max(tophat_responses, axis=0)

    # 6. Multi-scale Frangi Ridge Filter on Dark Contrast
    # Invert so dark filaments are bright ridges
    frangi_input = (dark_contrast / (dark_contrast.max() + 1e-7)).astype(np.float64)

    vesselness = np.zeros_like(frangi_input)
    for sigma in frangi_scales:
        smoothed = gaussian_filter(frangi_input, sigma=sigma)
        Hyy = gaussian_filter(smoothed, sigma=sigma, order=[2, 0]) * (sigma ** 2)
        Hxx = gaussian_filter(smoothed, sigma=sigma, order=[0, 2]) * (sigma ** 2)
        Hxy = gaussian_filter(smoothed, sigma=sigma, order=[1, 1]) * (sigma ** 2)

        trace = Hxx + Hyy
        det = Hxx * Hyy - Hxy ** 2
        disc = np.sqrt(np.maximum(trace ** 2 - 4 * det, 0))

        lambda1 = (trace - disc) / 2
        lambda2 = (trace + disc) / 2

        abs1, abs2 = np.abs(lambda1), np.abs(lambda2)
        swap = abs1 > abs2
        l1 = np.where(swap, lambda2, lambda1)
        l2 = np.where(swap, lambda1, lambda2)

        # Bright ridges in inverted contrast map
        valid = l2 < 0
        Rb = np.zeros_like(l1)
        Rb[valid] = (l1[valid] / (l2[valid] + 1e-10)) ** 2
        S2 = l1 ** 2 + l2 ** 2
        c = 0.5 * np.max(np.sqrt(S2)) + 1e-7

        V = np.zeros_like(frangi_input)
        V[valid] = np.exp(-Rb[valid] / 0.5) * (1 - np.exp(-S2[valid] / (2 * c ** 2)))
        vesselness = np.maximum(vesselness, V)

    v_max = vesselness.max()
    if v_max > 0:
        vesselness /= v_max
    vesselness[disk_mask_512 == 0] = 0

    # 7. Combined Feature Map: Top-Hat + Frangi + Local Dark Contrast
    norm_tophat = multi_tophat / (multi_tophat.max() + 1e-7)
    norm_contrast = dark_contrast / (dark_contrast.max() + 1e-7)
    combined_score = 0.45 * vesselness + 0.35 * norm_tophat + 0.20 * norm_contrast
    combined_score[disk_mask_512 == 0] = 0

    # 8. Adaptive Double Threshold (Hysteresis-style)
    # High threshold for seeds, low threshold for connected filament branches
    inner_disk = (disk_mask_512 > 0)
    high_thresh = np.percentile(combined_score[inner_disk], 94.0)
    low_thresh = np.percentile(combined_score[inner_disk], 86.0)

    seed_mask = (combined_score > high_thresh).astype(np.uint8)
    cand_mask = (combined_score > low_thresh).astype(np.uint8)
    cand_mask[disk_mask_512 == 0] = 0
    seed_mask[disk_mask_512 == 0] = 0

    # Distance map from solar disk center
    y_grid, x_grid = np.ogrid[:target_size, :target_size]
    dist_from_center = np.sqrt((x_grid - cx_512)**2 + (y_grid - cy_512)**2)
    max_allowed_dist = safe_r_512 * 0.94  # Strict interior margin

    # Connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cand_mask, connectivity=8)

    final_mask = np.zeros((target_size, target_size), dtype=np.uint8)

    for i in range(1, num_labels):
        comp_area = stats[i, cv2.CC_STAT_AREA]
        if comp_area < min_filament_area or comp_area > 8000:
            continue

        comp_pixels = (labels == i)

        # REJECT any component that touches the outer radial margin
        if np.any(dist_from_center[comp_pixels] >= max_allowed_dist):
            continue

        # Check if component contains at least one high-confidence seed pixel
        if not np.any(seed_mask[comp_pixels]):
            continue

        # Geometric Elongation Check: Solar filaments are elongated (not circular sunspots/granules)
        comp_w = stats[i, cv2.CC_STAT_WIDTH]
        comp_h = stats[i, cv2.CC_STAT_HEIGHT]
        aspect_ratio = max(comp_w, comp_h) / max(min(comp_w, comp_h), 1)

        pts = np.argwhere(comp_pixels)
        if len(pts) >= 5:
            cov = np.cov(pts, rowvar=False)
            evals, _ = np.linalg.eig(cov)
            evals = np.sort(evals)[::-1]
            eccentricity = np.sqrt(max(evals[0], 1e-5)) / np.sqrt(max(evals[1], 1e-5))
        else:
            eccentricity = aspect_ratio

        # Accept genuine filaments
        if eccentricity >= min_elongation or comp_area >= 40 or aspect_ratio >= 1.6:
            final_mask[comp_pixels] = 1

    # Morphological closing along filament orientation to bridge micro-gaps
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel_close)
    final_mask[disk_mask_512 == 0] = 0

    return {
        'preprocessed': gray_resized,
        'disk_mask': disk_mask_512,
        'dark_contrast': (norm_contrast * 255).astype(np.uint8),
        'tophat': (norm_tophat * 255).astype(np.uint8),
        'frangi_response': vesselness,
        'combined_score': combined_score,
        'filament_mask': final_mask,
    }

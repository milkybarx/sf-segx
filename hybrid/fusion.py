"""
Hybrid Fusion
=============
Combine U-Net and Frangi predictions for improved filament segmentation.
"""

import numpy as np
from typing import Dict, List


def fuse_predictions(unet_prob: np.ndarray, frangi_prob: np.ndarray,
                      alpha: float = 0.5) -> np.ndarray:
    """
    Weighted fusion of U-Net and Frangi probability maps.

    final = alpha * unet_prob + (1-alpha) * frangi_prob

    Args:
        unet_prob: U-Net probability map [0, 1]
        frangi_prob: Frangi response map [0, 1]
        alpha: Weight for U-Net (0 = Frangi only, 1 = U-Net only)

    Returns:
        Fused probability map [0, 1]
    """
    # Ensure same shape
    if unet_prob.shape != frangi_prob.shape:
        import cv2
        frangi_prob = cv2.resize(frangi_prob, (unet_prob.shape[1], unet_prob.shape[0]),
                                  interpolation=cv2.INTER_LINEAR)

    # Normalize frangi to [0, 1] if needed
    fmax = frangi_prob.max()
    if fmax > 0:
        frangi_norm = frangi_prob / fmax
    else:
        frangi_norm = frangi_prob

    fused = alpha * unet_prob + (1 - alpha) * frangi_norm
    return np.clip(fused, 0, 1)


def intersection_fusion(unet_mask: np.ndarray, frangi_mask: np.ndarray) -> np.ndarray:
    """
    Intersection-based fusion: only keep regions detected by BOTH methods.
    High precision, lower recall.
    """
    return (unet_mask & frangi_mask).astype(np.uint8)


def union_fusion(unet_mask: np.ndarray, frangi_mask: np.ndarray) -> np.ndarray:
    """
    Union-based fusion: keep regions detected by EITHER method.
    Higher recall, lower precision.
    """
    return (unet_mask | frangi_mask).astype(np.uint8)


def sweep_alpha(unet_probs: List[np.ndarray], frangi_probs: List[np.ndarray],
                gt_masks: List[np.ndarray],
                alphas: List[float] = [0.0, 0.25, 0.5, 0.75, 1.0]) -> Dict[float, Dict]:
    """
    Sweep fusion alpha values and report metrics for each.

    Args:
        unet_probs: List of U-Net probability maps
        frangi_probs: List of Frangi probability maps
        gt_masks: List of ground truth masks
        alphas: List of alpha values to test

    Returns:
        Dict mapping alpha -> metrics dict
    """
    from training.metrics import compute_metrics_numpy

    results = {}
    for alpha in alphas:
        all_metrics = []
        for unet_p, frangi_p, gt in zip(unet_probs, frangi_probs, gt_masks):
            fused = fuse_predictions(unet_p, frangi_p, alpha)
            fused_mask = (fused > 0.5).astype(np.uint8)
            metrics = compute_metrics_numpy(fused_mask, gt)
            all_metrics.append(metrics)

        # Average metrics
        avg = {}
        for key in all_metrics[0].keys():
            avg[key] = np.mean([m[key] for m in all_metrics])

        results[alpha] = avg
        print(f"  Alpha {alpha:.2f}: Dice={avg['dice']:.4f} IoU={avg['iou']:.4f} "
              f"P={avg['precision']:.4f} R={avg['recall']:.4f}")

    return results

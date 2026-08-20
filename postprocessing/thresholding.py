"""Probability thresholding and validation-only threshold selection."""
from typing import Dict, Iterable, Optional, Tuple
import numpy as np


def probability_to_mask(probability: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Convert a finite probability map to a uint8 binary mask."""
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    values = np.nan_to_num(probability, nan=0.0, posinf=1.0, neginf=0.0)
    return (values >= threshold).astype(np.uint8)


def segmentation_metrics(prediction: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    """Return Dice, IoU, precision, and recall for binary masks."""
    pred = prediction.astype(bool)
    truth = target.astype(bool)
    tp = float(np.logical_and(pred, truth).sum())
    fp = float(np.logical_and(pred, ~truth).sum())
    fn = float(np.logical_and(~pred, truth).sum())
    return {"dice": 2 * tp / (2 * tp + fp + fn + 1e-8),
            "iou": tp / (tp + fp + fn + 1e-8),
            "precision": tp / (tp + fp + 1e-8),
            "recall": tp / (tp + fn + 1e-8)}


def threshold_sweep(probability: np.ndarray, target: np.ndarray,
                    thresholds: Optional[Iterable[float]] = None) -> Tuple[float, list]:
    """Evaluate thresholds on validation data and return the best Dice threshold."""
    values = list(thresholds if thresholds is not None else np.arange(0.20, 0.751, 0.05))
    scores = [{"threshold": float(t), **segmentation_metrics(probability_to_mask(probability, t), target)}
              for t in values]
    best = max(scores, key=lambda item: item["dice"])
    return float(best["threshold"]), scores

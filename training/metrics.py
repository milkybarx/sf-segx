"""
Segmentation Metrics
====================
Dice, IoU, Precision, Recall for binary segmentation evaluation.
"""

import torch
import numpy as np
from typing import Dict


def dice_coefficient(pred: torch.Tensor, target: torch.Tensor,
                      threshold: float = 0.5, smooth: float = 1.0) -> float:
    """Compute Dice coefficient (F1 score for segmentation)."""
    pred_binary = (torch.sigmoid(pred) > threshold).float()
    target_flat = target.view(-1)
    pred_flat = pred_binary.view(-1)

    intersection = (pred_flat * target_flat).sum()
    dice = (2.0 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
    return dice.item()


def iou_score(pred: torch.Tensor, target: torch.Tensor,
               threshold: float = 0.5, smooth: float = 1.0) -> float:
    """Compute Intersection over Union."""
    pred_binary = (torch.sigmoid(pred) > threshold).float()
    target_flat = target.view(-1)
    pred_flat = pred_binary.view(-1)

    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou.item()


def precision_score(pred: torch.Tensor, target: torch.Tensor,
                     threshold: float = 0.5, smooth: float = 1e-7) -> float:
    """Compute pixel-level precision."""
    pred_binary = (torch.sigmoid(pred) > threshold).float()
    tp = (pred_binary * target).sum()
    fp = (pred_binary * (1 - target)).sum()
    precision = (tp + smooth) / (tp + fp + smooth)
    return precision.item()


def recall_score(pred: torch.Tensor, target: torch.Tensor,
                  threshold: float = 0.5, smooth: float = 1e-7) -> float:
    """Compute pixel-level recall."""
    pred_binary = (torch.sigmoid(pred) > threshold).float()
    tp = (pred_binary * target).sum()
    fn = ((1 - pred_binary) * target).sum()
    recall = (tp + smooth) / (tp + fn + smooth)
    return recall.item()


def compute_all_metrics(pred: torch.Tensor, target: torch.Tensor,
                         threshold: float = 0.5) -> Dict[str, float]:
    """Compute all segmentation metrics."""
    return {
        'dice': dice_coefficient(pred, target, threshold),
        'iou': iou_score(pred, target, threshold),
        'precision': precision_score(pred, target, threshold),
        'recall': recall_score(pred, target, threshold),
    }


def compute_metrics_numpy(pred_mask: np.ndarray, gt_mask: np.ndarray,
                            smooth: float = 1e-7) -> Dict[str, float]:
    """Compute metrics from numpy binary masks."""
    pred = pred_mask.astype(np.float32).flatten()
    gt = gt_mask.astype(np.float32).flatten()

    tp = (pred * gt).sum()
    fp = (pred * (1 - gt)).sum()
    fn = ((1 - pred) * gt).sum()

    intersection = tp
    union = tp + fp + fn

    dice = (2 * intersection + smooth) / (pred.sum() + gt.sum() + smooth)
    iou = (intersection + smooth) / (union + smooth)
    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)

    return {
        'dice': float(dice),
        'iou': float(iou),
        'precision': float(precision),
        'recall': float(recall),
    }

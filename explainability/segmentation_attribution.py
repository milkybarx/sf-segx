"""Mask2Former-compatible dense segmentation attribution."""
from typing import Optional
import cv2
import numpy as np
import torch


def segmentation_attribution(model: torch.nn.Module, preprocessed: np.ndarray,
                              target_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Compute input-gradient saliency for the model's dense filament segmentation.

    This targets the custom Mask2Former dense logit output, rather than applying a
    classification Grad-CAM hook to an unrelated CNN layer.
    """
    tensor = torch.from_numpy(preprocessed.astype(np.float32) / 255.0)[None, None]
    tensor.requires_grad_(True)
    model.eval()
    model.zero_grad(set_to_none=True)
    logits = model(tensor)
    weights = torch.from_numpy(target_mask.astype(np.float32))[None, None] if target_mask is not None else torch.sigmoid(logits).detach()
    objective = (logits * weights).mean()
    objective.backward()
    attribution = tensor.grad.detach().abs().squeeze().cpu().numpy()
    attribution -= attribution.min()
    return (attribution / max(float(attribution.max()), 1e-8)).astype(np.float32)


def attribution_overlay(image: np.ndarray, attribution: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Return an RGB heatmap overlay."""
    base = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) if image.ndim == 2 else image.copy()
    heat = cv2.applyColorMap((np.clip(attribution, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    if heat.shape[:2] != base.shape[:2]:
        heat = cv2.resize(heat, (base.shape[1], base.shape[0]))
    return cv2.addWeighted(base, 1 - alpha, heat, alpha, 0)

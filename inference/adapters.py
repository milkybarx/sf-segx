"""Common model-output interface for all compatible segmentation models."""
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import cv2
import numpy as np
import torch


@dataclass
class StandardizedPrediction:
    """Model-independent segmentation result consumed by Phase 2."""
    mask: np.ndarray
    probability: np.ndarray
    confidence: Optional[float]
    model_name: str
    model_checkpoint: Optional[str]
    image: np.ndarray
    preprocessed: np.ndarray
    disk_mask: np.ndarray
    logits: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class SegmentationModelAdapter:
    """Adapter contract that hides model-specific loading and output formats."""
    def __init__(self, model_name: str, model=None):
        self.model_name = model_name
        self.model = model

    @property
    def model_checkpoint(self) -> Optional[str]:
        from model_hub import checkpoint_path_for
        return checkpoint_path_for(self.model_name)

    @property
    def supports_explainability(self) -> bool:
        # "mask2former"/"mask2former_scratch" are stale arch names from an earlier version
        # of this repo's model_hub.py -- the currently-registered Mask2Former is
        # "mask2former_phase3" (see model_hub.EXTERNAL_MODELS).
        return self.model_name in {"mask2former_phase3"}

    def predict(self, image: np.ndarray, threshold: float = 0.5) -> StandardizedPrediction:
        """Return HxW arrays regardless of the selected model architecture."""
        from model_hub import DISPLAY_SIZE, EXTERNAL_MODELS, get_model, run_inference
        if self.model_name == "mask2former":
            from inference.mask2former import run_mask2former_inference
            result = run_mask2former_inference(image, self.model, threshold)
            probability = np.asarray(result.probability, dtype=np.float32)
            mask = (probability >= threshold).astype(np.uint8)
            confidence = float(probability[mask > 0].mean()) if np.any(mask) else float(probability.max())
            return StandardizedPrediction(mask=mask, probability=probability, confidence=confidence,
                                          model_name=self.model_name, model_checkpoint=self.model_checkpoint,
                                          image=result.image, preprocessed=result.preprocessed,
                                          disk_mask=result.disk_mask, logits=result.logits,
                                          metadata={"native_output": "dense semantic logits", "threshold": threshold})

        if self.model is None:
            self.model, _ = get_model(self.model_name)
        small, disk_small, probability_small, mask_small = run_inference(image, self.model_name, threshold)
        probability = cv2.resize(np.asarray(probability_small, dtype=np.float32),
                                 (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
        disk_mask = cv2.resize(disk_small.astype(np.uint8), (image.shape[1], image.shape[0]),
                               interpolation=cv2.INTER_NEAREST).astype(bool)
        mask = cv2.resize(mask_small.astype(np.uint8), (image.shape[1], image.shape[0]),
                          interpolation=cv2.INTER_NEAREST)
        probability[~disk_mask] = 0.0
        mask[~disk_mask] = 0
        confidence = float(probability[mask > 0].mean()) if np.any(mask) else float(probability.max())
        return StandardizedPrediction(mask=mask, probability=probability, confidence=confidence,
                                      model_name=self.model_name, model_checkpoint=self.model_checkpoint,
                                      image=image, preprocessed=small, disk_mask=disk_mask,
                                      metadata={"native_output": "model_hub normalized probability", "threshold": threshold,
                                                "display_shape": (DISPLAY_SIZE, DISPLAY_SIZE),
                                                "external_kind": EXTERNAL_MODELS.get(self.model_name, {}).get("kind")})


def get_segmentation_adapter(model_name: str, model=None) -> SegmentationModelAdapter:
    """Create an adapter for any model registered by model_hub."""
    return SegmentationModelAdapter(model_name, model)

"""Model-aware explainability dispatch for Phase 2."""
from typing import Optional
import numpy as np


def generate_explanation(model, image: np.ndarray, prediction, model_type: str) -> Optional[np.ndarray]:
    """Generate attribution only for architectures with a compatible implementation."""
    if model_type in {"mask2former", "mask2former_scratch"}:
        from explainability.segmentation_attribution import segmentation_attribution
        target = np.asarray(prediction.mask, dtype=np.uint8)
        if target.shape != prediction.preprocessed.shape[:2]:
            import cv2
            target = cv2.resize(target, (prediction.preprocessed.shape[1], prediction.preprocessed.shape[0]),
                                interpolation=cv2.INTER_NEAREST)
        return segmentation_attribution(model, prediction.preprocessed, target)
    return None

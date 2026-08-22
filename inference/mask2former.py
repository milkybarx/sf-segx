"""Model-specific inference adapter for the repository Mask2Former checkpoint."""
from dataclasses import dataclass
from typing import Any, Optional
import cv2
import numpy as np
import torch

@dataclass
class Mask2FormerInference:
    """Dense semantic segmentation outputs at the original image dimensions."""
    image: np.ndarray
    preprocessed: np.ndarray
    probability: np.ndarray
    mask: np.ndarray
    disk_mask: np.ndarray
    logits: Optional[np.ndarray]
    image_shape: tuple


def run_mask2former_inference(image: np.ndarray, model: Optional[torch.nn.Module] = None,
                              threshold: float = 0.5, device: Optional[torch.device] = None) -> Mask2FormerInference:
    """Run the custom grayscale Mask2Former with its training-time preprocessing."""
    from model_hub import DISPLAY_SIZE, get_ext_preprocessor, get_model
    if model is None:
        model, _ = get_model("mask2former")
    if model is None:
        raise FileNotFoundError("Mask2Former checkpoint could not be loaded")
    preprocessor = get_ext_preprocessor()
    prepared = preprocessor.preprocess(image, return_intermediates=True)
    small = cv2.resize(prepared["preprocessed"], (DISPLAY_SIZE, DISPLAY_SIZE), interpolation=cv2.INTER_AREA)
    disk_small = cv2.resize(prepared["disk_mask"].astype(np.uint8), (DISPLAY_SIZE, DISPLAY_SIZE), interpolation=cv2.INTER_NEAREST).astype(bool)
    target_device = device or next(model.parameters()).device
    tensor = torch.from_numpy(small.astype(np.float32) / 255.0)[None, None].to(target_device)
    model.eval()
    with torch.no_grad():
        logits_tensor = model(tensor)
        logits_small = logits_tensor.squeeze().detach().cpu().numpy().astype(np.float32)
        probability_small = torch.sigmoid(logits_tensor).squeeze().detach().cpu().numpy()
    probability_small = np.nan_to_num(probability_small, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)
    probability_small[~disk_small] = 0.0
    probability = cv2.resize(probability_small, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
    disk_mask = cv2.resize(disk_small.astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
    probability[~disk_mask] = 0.0
    logits = cv2.resize(logits_small, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
    return Mask2FormerInference(image=image, preprocessed=small, probability=probability,
                                mask=(probability >= threshold).astype(np.uint8), disk_mask=disk_mask,
                                logits=logits, image_shape=image.shape[:2])

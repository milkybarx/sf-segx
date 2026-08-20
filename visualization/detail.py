"""High-magnification presentation helpers for the Phase 2 filament inspector."""
from pathlib import Path
from typing import Dict, Optional
import json
import cv2
import numpy as np


def crop_filament(image: np.ndarray, filament: Dict, padding: int = 30):
    """Crop the original image around one existing filament bounding box."""
    if padding < 0:
        raise ValueError("padding must be non-negative")
    bbox = filament["bbox"]
    x_min = bbox.get("x_min", bbox.get("x", 0))
    y_min = bbox.get("y_min", bbox.get("y", 0))
    x_max = bbox.get("x_max", x_min + bbox.get("width", 0))
    y_max = bbox.get("y_max", y_min + bbox.get("height", 0))
    
    x0 = max(0, int(x_min) - padding)
    y0 = max(0, int(y_min) - padding)
    x1 = min(image.shape[1], int(x_max) + padding)
    y1 = min(image.shape[0], int(y_max) + padding)
    return image[y0:y1, x0:x1].copy(), (x0, y0, x1, y1)


def super_resolve_crop(crop: np.ndarray, method: str = "Lanczos (Current)", scale: int = 2,
                       model=None, device="cpu") -> np.ndarray:
    """Upscale visualization crop with specified method (interpolation or AI-SR)."""
    if scale < 1:
        raise ValueError("scale must be positive")
    if scale == 1 or method == "OFF":
        return crop.copy()
    
    if method in ["Lanczos (Current)", "Bicubic"]:
        interp = cv2.INTER_LANCZOS4 if method == "Lanczos (Current)" else cv2.INTER_CUBIC
        return cv2.resize(crop, (crop.shape[1] * scale, crop.shape[0] * scale), interpolation=interp)
    
    if model is not None:
        import torch
        with torch.inference_mode():
            lr_float = crop.astype(np.float32) / 255.0
            if lr_float.ndim == 2:
                lr_tensor = torch.from_numpy(lr_float).unsqueeze(0).unsqueeze(0).to(device)
            else:
                lr_tensor = torch.from_numpy(lr_float).permute(2, 0, 1).unsqueeze(0).to(device)
                
            sr_tensor = model(lr_tensor)
            sr_np = sr_tensor.squeeze().cpu().numpy()
            
            if sr_np.ndim == 3:
                sr_np = sr_np.transpose(1, 2, 0)
            
            return (sr_np * 255.0).clip(0, 255).astype(np.uint8)
            
    # Fallback to Lanczos if AI model is not provided
    return cv2.resize(crop, (crop.shape[1] * scale, crop.shape[0] * scale), interpolation=cv2.INTER_LANCZOS4)


def selected_overlay(crop: np.ndarray, filament: Dict, labels: np.ndarray,
                     crop_bounds: tuple, show_mask: bool = True,
                     show_skeleton: bool = True, show_bbox: bool = True,
                     show_labels: bool = True,
                     attribution: Optional[np.ndarray] = None,
                     show_attribution: bool = False) -> np.ndarray:
    """Render selected-filament overlays without changing any scientific arrays."""
    x0, y0, x1, y1 = crop_bounds
    display = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
    component = (labels[y0:y1, x0:x1] == filament.get("component_id", -1))
    if show_mask:
        color = np.zeros_like(display)
        color[component] = (235, 45, 65)
        display = cv2.addWeighted(display, 0.72, color, 0.28, 0)
        contours, _ = cv2.findContours(component.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(display, contours, -1, (255, 190, 0), 1)
    if show_attribution and attribution is not None:
        if attribution.shape[:2] != labels.shape[:2]:
            attribution = cv2.resize(attribution, (labels.shape[1], labels.shape[0]), interpolation=cv2.INTER_LINEAR)
        heat = cv2.applyColorMap((np.clip(attribution[y0:y1, x0:x1], 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
        display = cv2.addWeighted(display, 0.68, heat, 0.32, 0)
    local = dict(filament)
    bbox = filament["bbox"]
    x_min = bbox.get("x_min", bbox.get("x", 0))
    y_min = bbox.get("y_min", bbox.get("y", 0))
    x_max = bbox.get("x_max", x_min + bbox.get("width", 0))
    y_max = bbox.get("y_max", y_min + bbox.get("height", 0))
    
    local["bbox"] = {"x_min": int(x_min) - x0, "y_min": int(y_min) - y0,
                     "x_max": int(x_max) - x0, "y_max": int(y_max) - y0}
    local["image_width"], local["image_height"] = display.shape[1], display.shape[0]
    if filament.get("skeleton_mask") is not None:
        local["skeleton_mask"] = filament["skeleton_mask"][y0:y1, x0:x1]
    from visualization.phase2 import _instance_panel
    if show_skeleton or show_bbox or show_labels:
        display = _instance_panel(display, [local], skeleton=show_skeleton,
                                   draw_boxes=show_bbox, draw_labels=show_labels)
    return display


def detail_record(filament: Dict) -> Dict:
    """Return all existing calculated fields in a JSON-safe detail record."""
    record = {}
    for key, value in filament.items():
        if key in {"skeleton_mask", "component_mask"}:
            continue
        if isinstance(value, np.generic):
            value = value.item()
        record[key] = value
    return record


def save_detail_artifacts(output_dir: str | Path, original: np.ndarray,
                          enhanced: Optional[np.ndarray], overlay: np.ndarray,
                          filament: Dict) -> Dict[str, Path]:
    """Save selected-filament crops, overlay, and JSON without altering the overview."""
    directory = Path(output_dir) / f"filament_{int(filament['filament_id']):03d}"
    directory.mkdir(parents=True, exist_ok=True)
    paths = {"original_crop": directory / "original_crop.png",
             "overlay": directory / "overlay.png",
             "filament_json": directory / "filament.json"}
    cv2.imwrite(str(paths["original_crop"]), original)
    cv2.imwrite(str(paths["overlay"]), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    if enhanced is not None:
        paths["enhanced_crop"] = directory / "enhanced_crop.png"
        cv2.imwrite(str(paths["enhanced_crop"]), enhanced)
    paths["filament_json"].write_text(json.dumps(detail_record(filament), indent=2, default=str), encoding="utf-8")
    return paths

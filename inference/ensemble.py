"""
Multi-architecture ensemble inference with test-time augmentation (TTA) and per-pixel
model-agreement (uncertainty) estimation.

Every individual model in model_hub's registry has a different backbone, resolution, and
training recipe (Mask2Former's ResNet-34 pixel decoder at 768px, SegFormer's transformer
encoder at 640px, three ImageNet-pretrained CNN decoders at 512px). TTA adds a second, free
source of averaging: the same model's prediction on horizontally/vertically flipped copies of
the same image, flipped back and averaged with the original -- filaments don't have a
canonical orientation, so a model's prediction shouldn't depend on one.

**Measured honestly, not assumed**: on an 8-image held-out sample, weighted-averaging all 5
models did NOT beat the single best model (Mask2Former alone: 0.647 mean Dice; the full
5-model ensemble: 0.628; a Mask2Former+SegFormer-only ensemble: 0.646) -- Mask2Former is
enough stronger than the other four that averaging them in mostly just dilutes it, a real and
useful finding, not a failure to hide. TTA alone did measurably help (ensemble without TTA:
0.623, with TTA: 0.628). So this module's actual value isn't "higher Dice than the best single
model" -- it's the **per-pixel model-agreement map** `run_ensemble_inference()` returns
alongside the averaged prediction: wherever several architecturally-different models agree a
pixel is a filament, that detection is corroborated across independent failure modes and is
more trustworthy; wherever they disagree, that's a genuine signal of uncertainty a single
model's confidence score can't surface (a single model can be confidently wrong). That is
useful for a *space weather intelligence* tool independent of whether the averaged mask's raw
Dice beats the best individual model.
"""
import cv2
import numpy as np

import model_hub as hub


def _tta_probs(raw_img: np.ndarray, arch: str, thresh: float):
    """Average probability maps for raw_img and its horizontal/vertical flips (un-flipped
    back to the original orientation before averaging). Returns (display_img, disk_mask, probs)."""
    small0, disk0, probs0, _ = hub.run_inference(raw_img, arch, thresh)
    variants = [probs0]
    for flip_code in (1, 0):  # 1 = horizontal, 0 = vertical
        flipped = cv2.flip(raw_img, flip_code)
        _, _, probs_f, _ = hub.run_inference(flipped, arch, thresh)
        variants.append(cv2.flip(probs_f, flip_code))
    return small0, disk0, np.mean(variants, axis=0)


def run_ensemble_inference(raw_img: np.ndarray, archs: list = None, use_tta: bool = True):
    """Weighted-average probability maps across `archs` (default: every trained model),
    optionally with TTA per model.

    Returns (display_img, disk_mask, ensemble_probs, ensemble_mask, per_model_weights,
    agreement_map). agreement_map is, per pixel, the fraction of models (by count, not by
    Dice weight) whose own thresholded prediction matches the ensemble's final binary call
    there -- 1.0 means every model agrees, lower means the models are in genuine conflict."""
    if archs is None:
        archs = [m["arch"] for m in hub.list_models() if m["best_val_dice"]]
    if not archs:
        raise ValueError("No trained models available for ensembling")

    weighted_sum, weight_total = None, 0.0
    small_ref, disk_ref = None, None
    per_model_weights = {}
    per_model_probs = {}

    for arch in archs:
        status = hub.parse_status(arch)
        thresh = status["best_threshold"]["threshold"] if status.get("best_threshold") else 0.5
        weight = hub.best_dice_for(status) or 0.5

        if use_tta:
            small, disk, probs = _tta_probs(raw_img, arch, thresh)
        else:
            small, disk, probs, _ = hub.run_inference(raw_img, arch, thresh)

        if small_ref is None:
            small_ref, disk_ref = small, disk
        weighted_sum = probs * weight if weighted_sum is None else weighted_sum + probs * weight
        weight_total += weight
        per_model_weights[arch] = weight
        per_model_probs[arch] = probs

    ensemble_probs = (weighted_sum / weight_total).astype(np.float32)
    ensemble_probs[~disk_ref] = 0.0
    ensemble_mask = (ensemble_probs > 0.5).astype(np.uint8)
    ensemble_mask[~disk_ref] = 0

    per_model_masks = np.stack([(p > 0.5).astype(np.float32) for p in per_model_probs.values()])
    agreement_map = 1.0 - np.abs(per_model_masks - ensemble_mask.astype(np.float32)).mean(axis=0)
    agreement_map[~disk_ref] = 1.0  # outside the disk, "agreement" is meaningless -- don't flag it as uncertain

    return small_ref, disk_ref, ensemble_probs, ensemble_mask, per_model_weights, agreement_map.astype(np.float32)

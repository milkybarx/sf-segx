"""Model-aware explainability dispatch for Phase 2."""
from typing import Optional
import cv2
import numpy as np


def generate_explanation(model, image: np.ndarray, prediction, model_type: str) -> Optional[np.ndarray]:
    """Generate attribution only for architectures with a compatible implementation.

    "mask2former"/"mask2former_scratch" are stale arch names from an earlier version of
    this repo -- the currently-registered Mask2Former is "mask2former_phase3".
    """
    if model_type in {"mask2former_phase3"}:
        from explainability.segmentation_attribution import segmentation_attribution
        import model_hub as hub

        # prediction.preprocessed is DISPLAY_SIZE (512) for the dashboard's shared preview,
        # not the model's own trained resolution (768 for mask2former_phase3) -- running
        # the model on the wrong resolution here would silently compute an attribution map
        # that doesn't correspond to what actually produced prediction.mask. Recompute the
        # model's own native-resolution input the same way model_hub.run_inference() does.
        spec = hub.EXTERNAL_MODELS.get(model_type, {})
        res = spec.get("resolution", hub.DISPLAY_SIZE)
        result = hub.get_ext_preprocessor().preprocess(image, return_intermediates=True)
        model_in = cv2.resize(result["preprocessed"], (res, res), interpolation=cv2.INTER_AREA)

        target = cv2.resize(np.asarray(prediction.mask, dtype=np.uint8), (res, res),
                            interpolation=cv2.INTER_NEAREST)
        attribution = segmentation_attribution(model, model_in, target)

        display_shape = prediction.preprocessed.shape[:2]
        if attribution.shape != display_shape:
            attribution = cv2.resize(attribution, (display_shape[1], display_shape[0]))
        return attribution
    return None

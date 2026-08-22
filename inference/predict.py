"""
Inference Pipeline
==================
Single-image inference for solar filament segmentation.
Supports U-Net, Frangi, and hybrid predictions.
Gracefully handles environments before deep-learning model training.
"""

import os
import sys
import time
import numpy as np
import cv2
import yaml
from typing import Dict, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.solar_preprocessor import SolarPreprocessor
from classical.frangi import FrangiPipeline
from hybrid.fusion import fuse_predictions


def try_load_model(checkpoint_path: str, config: dict):
    """Attempt to load trained Mask2Former or U-Net model if torch and checkpoint are present."""
    try:
        import torch
        if not os.path.exists(checkpoint_path):
            return None, None
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        saved_config = checkpoint.get('config', config)
        model_name = saved_config.get('model', {}).get('name', 'mask2former').lower()
        
        if model_name == 'mask2former':
            from models.mask2former import build_mask2former
            model = build_mask2former(saved_config.get('model', {}))
        else:
            from models.unet import build_unet
            model = build_unet(saved_config.get('model', {}))
            
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device).eval()
        print(f"Loaded {model_name.upper()} model from {checkpoint_path} on {device}")
        return model, device
    except Exception as e:
        print(f"DL Model not loaded ({e}). Running in Classical CV mode.")
        return None, None


class SolarFilamentPredictor:
    """Complete inference pipeline for solar filament segmentation."""

    def __init__(self, checkpoint_path: Optional[str] = None, config_path: Optional[str] = None):
        # Load config
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'configs', 'default_config.yaml'
            )
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = {'data': {'image_size': 512}}

        self.image_size = self.config.get('data', {}).get('image_size', 512)
        self.preprocessor = SolarPreprocessor(target_size=self.image_size)

        # Setup Frangi pipeline
        frangi_cfg = self.config.get('frangi', {})
        self.frangi = FrangiPipeline(
            scales=frangi_cfg.get('scales', [1, 2, 3, 5, 8]),
            alpha=frangi_cfg.get('alpha', 0.5),
            beta=frangi_cfg.get('beta', 0.5),
            gamma=frangi_cfg.get('gamma', 15.0),
            threshold=frangi_cfg.get('threshold', 0.15),
            min_area=frangi_cfg.get('min_area', 25),
            max_area=frangi_cfg.get('max_area', 12000),
            target_size=self.image_size,
        )

        # Try to load deep learning model (Mask2Former / U-Net)
        if checkpoint_path is None:
            checkpoint_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'checkpoints', 'best_model.pth'
            )
        self.model, self.device = try_load_model(checkpoint_path, self.config)

    def predict_unet(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
        """Run U-Net prediction if available."""
        if self.model is None:
            return None, 0.0

        import torch
        preprocessed = self.preprocessor.preprocess_for_model(image)
        tensor = torch.from_numpy(preprocessed).unsqueeze(0).unsqueeze(0).to(self.device)

        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        start = time.time()

        with torch.no_grad():
            logits = self.model(tensor)
            prob = torch.sigmoid(logits)

        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        elapsed = time.time() - start

        prob_map = prob.squeeze().cpu().numpy()
        return prob_map, elapsed

    def predict_frangi(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Run Frangi pipeline."""
        return self.frangi.process_resized(image)

    def predict(self, image: np.ndarray, method: str = 'hybrid',
                fusion_alpha: float = 0.5) -> Dict[str, np.ndarray]:
        """
        Full prediction pipeline.
        Returns dict with all visualization outputs.
        """
        results = {}
        results['original'] = image.copy()

        # Preprocessing visualization
        preproc = self.preprocessor.preprocess(image, return_intermediates=True)
        results['preprocessed'] = preproc['preprocessed']
        results['inverted'] = preproc['inverted']

        # Frangi pipeline
        frangi_results = self.predict_frangi(image)
        results['frangi_response'] = frangi_results.get('frangi_response', np.zeros((self.image_size, self.image_size)))
        results['hessian_response'] = frangi_results.get('hessian_response', np.zeros((self.image_size, self.image_size)))
        results['frangi_mask'] = frangi_results.get('filament_mask', np.zeros((self.image_size, self.image_size)))
        frangi_prob = frangi_results.get('frangi_probability', np.zeros((self.image_size, self.image_size)))
        results['frangi_probability'] = frangi_prob

        # U-Net prediction
        unet_prob, inference_time = self.predict_unet(image)
        results['inference_time'] = inference_time

        if unet_prob is not None:
            results['unet_probability'] = unet_prob
            results['unet_mask'] = (unet_prob > 0.5).astype(np.uint8)

            fused_prob = fuse_predictions(unet_prob, frangi_prob, alpha=fusion_alpha)
            results['hybrid_probability'] = fused_prob
            results['hybrid_mask'] = (fused_prob > 0.5).astype(np.uint8)

            if method.lower() in ['unet', 'mask2former', 'deeplearning']:
                results['final_mask'] = results['unet_mask']
                results['final_probability'] = results['unet_probability']
            elif method.lower() == 'frangi':
                results['final_mask'] = results['frangi_mask']
                results['final_probability'] = results['frangi_probability']
            else:  # hybrid
                results['final_mask'] = results['hybrid_mask']
                results['final_probability'] = results['hybrid_probability']
        else:
            # Fallback to Frangi
            results['unet_probability'] = frangi_prob
            results['unet_mask'] = results['frangi_mask']
            results['hybrid_probability'] = frangi_prob
            results['hybrid_mask'] = results['frangi_mask']
            results['final_mask'] = results['frangi_mask']
            results['final_probability'] = results['frangi_probability']

        # Create overlay
        results['overlay'] = create_overlay(image, results['final_mask'], self.image_size)

        return results


def create_overlay(image: np.ndarray, mask: np.ndarray,
                    target_size: int = 512, color: tuple = (0, 0, 255),
                    alpha: float = 0.4) -> np.ndarray:
    """Create semi-transparent overlay of detected filaments on original image."""
    resized = cv2.resize(image, (target_size, target_size))
    if len(resized.shape) == 2:
        resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)

    overlay = resized.copy()
    mask_bool = mask.astype(bool)
    overlay[mask_bool] = color

    result = cv2.addWeighted(resized, 1 - alpha, overlay, alpha, 0)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, (0, 255, 255), 1)

    return result


# Backward-compatible alias
SingleImagePredictor = SolarFilamentPredictor


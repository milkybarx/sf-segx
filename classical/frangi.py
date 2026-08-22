"""
Frangi Filter & Ridge Detection Pipeline
=========================================
Classical computer vision pipeline for solar filament detection using
Multi-Scale Frangi vesselness, Black Top-Hat transform, and Hessian analysis.
"""

import numpy as np
import cv2
from typing import Dict, List, Optional
from classical.advanced_extractor import extract_filaments_advanced


class FrangiPipeline:
    """
    Complete classical filament detection pipeline.
    """

    def __init__(self,
                 scales: List[float] = [1.0, 1.8, 3.0, 5.0, 7.5],
                 alpha: float = 0.5,
                 beta: float = 0.5,
                 gamma: float = 15.0,
                 threshold: float = 0.15,
                 min_area: int = 20,
                 max_area: int = 15000,
                 target_size: int = 512):
        self.scales = scales
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.threshold = threshold
        self.min_area = min_area
        self.max_area = max_area
        self.target_size = target_size

    def process_resized(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Run advanced filament detection on solar image.
        Completely eliminates outer disk edge ring and captures all dark filaments.
        """
        res = extract_filaments_advanced(
            image=image,
            target_size=self.target_size,
            limb_margin=0.07,
            frangi_scales=self.scales,
            min_filament_area=self.min_area,
            min_elongation=1.5,
        )

        return {
            'original': cv2.resize(image, (self.target_size, self.target_size)) if len(image.shape)==3 else cv2.cvtColor(cv2.resize(image, (self.target_size, self.target_size)), cv2.COLOR_GRAY2BGR),
            'preprocessed': res['preprocessed'],
            'frangi_response': res['frangi_response'],
            'hessian_response': res['combined_score'],
            'filament_mask': res['filament_mask'],
            'frangi_probability': res['combined_score'],
        }

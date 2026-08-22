"""
Solar Image Preprocessor
========================
Handles grayscale conversion, normalization, denoising, contrast enhancement,
solar disk detection, limb darkening correction, and image inversion.

Designed for H-alpha full-disk solar observations from GONG/NSO.
"""

import numpy as np
import cv2
from typing import Tuple, Optional, Dict


class SolarPreprocessor:
    """Preprocessing pipeline for H-alpha solar images."""

    def __init__(self, target_size: int = 512, disk_shrink: float = 0.93):
        self.target_size = target_size
        # Fraction of the detected radius actually kept as "inside the disk" -- shrinking
        # further than the default excludes more of the limb edge but also more real
        # near-limb disk area. Kept as a constructor param (not touched here) specifically
        # so experiments can override it without changing the default every other caller
        # (including the currently-deployed Mask2Former checkpoint's own inference) relies on.
        self.disk_shrink = disk_shrink

    def to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Convert to grayscale if needed."""
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image.copy()

    def detect_solar_disk(self, gray: np.ndarray) -> Tuple[int, int, int]:
        """
        Detect the solar disk center and radius using thresholding + contour fitting.
        More robust than Hough circles for full-disk solar images.
        """
        # Threshold to separate disk from background
        _, binary = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)

        # Find largest contour (the solar disk)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            # Fallback: assume centered disk
            h, w = gray.shape
            return w // 2, h // 2, min(w, h) // 2 - 10

        largest = max(contours, key=cv2.contourArea)
        (cx, cy), radius = cv2.minEnclosingCircle(largest)
        return int(cx), int(cy), int(radius * self.disk_shrink)

    def create_disk_mask(self, shape: Tuple[int, int], cx: int, cy: int, radius: int) -> np.ndarray:
        """Create binary mask for the solar disk region."""
        mask = np.zeros(shape, dtype=np.uint8)
        cv2.circle(mask, (cx, cy), radius, 255, -1)
        return mask

    def correct_limb_darkening(self, gray: np.ndarray, cx: int, cy: int, radius: int) -> np.ndarray:
        """
        Correct for center-to-limb brightness variation in H-alpha images.
        Uses a radial polynomial model.
        """
        h, w = gray.shape
        y, x = np.ogrid[:h, :w]
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) / radius
        r = np.clip(r, 0, 1)

        # Limb darkening polynomial: I(r) = 1 - u*(1 - sqrt(1 - r^2))
        # Using a simple quadratic approximation
        mu = np.sqrt(np.maximum(1 - r ** 2, 0))
        u = 0.6  # Limb darkening coefficient for H-alpha
        correction = 1.0 / (1.0 - u * (1.0 - mu) + 1e-8)

        # Only apply within disk
        disk_mask = r < 0.98
        corrected = gray.astype(np.float64)
        corrected[disk_mask] = corrected[disk_mask] * correction[disk_mask]
        corrected = np.clip(corrected, 0, 255).astype(np.uint8)

        return corrected

    def normalize(self, gray: np.ndarray, disk_mask: np.ndarray) -> np.ndarray:
        """Normalize intensity within the solar disk region to [0, 255]."""
        result = gray.copy().astype(np.float64)
        masked = result[disk_mask > 0]
        if len(masked) == 0:
            return gray

        vmin, vmax = np.percentile(masked, [1, 99])
        if vmax - vmin < 1:
            return gray

        result = (result - vmin) / (vmax - vmin) * 255
        result = np.clip(result, 0, 255).astype(np.uint8)
        # Zero out background
        result[disk_mask == 0] = 0
        return result

    def denoise(self, gray: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        """Apply Gaussian blur for noise reduction."""
        ksize = int(sigma * 4) | 1  # Ensure odd
        return cv2.GaussianBlur(gray, (ksize, ksize), sigma)

    def enhance_contrast(self, gray: np.ndarray, clip_limit: float = 2.0,
                         tile_size: int = 8) -> np.ndarray:
        """Apply CLAHE for local contrast enhancement."""
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
        return clahe.apply(gray)

    def invert(self, gray: np.ndarray) -> np.ndarray:
        """Invert image so dark filaments become bright (for ridge detection)."""
        return 255 - gray

    def resize(self, image: np.ndarray, size: Optional[int] = None,
               interpolation=cv2.INTER_AREA) -> np.ndarray:
        """Resize image to target size."""
        size = size or self.target_size
        return cv2.resize(image, (size, size), interpolation=interpolation)

    def preprocess(self, image: np.ndarray, return_intermediates: bool = False) -> Dict[str, np.ndarray]:
        """
        Full preprocessing pipeline.

        Returns dict with at minimum 'preprocessed' key.
        If return_intermediates=True, also returns all intermediate steps.
        """
        intermediates = {}

        # Step 1: Grayscale
        gray = self.to_grayscale(image)
        intermediates['grayscale'] = gray.copy()

        # Step 2: Detect solar disk
        cx, cy, radius = self.detect_solar_disk(gray)
        disk_mask = self.create_disk_mask(gray.shape, cx, cy, radius)
        intermediates['disk_mask'] = disk_mask.copy()
        intermediates['disk_params'] = (cx, cy, radius)

        # Step 3: Limb darkening correction
        corrected = self.correct_limb_darkening(gray, cx, cy, radius)
        intermediates['limb_corrected'] = corrected.copy()

        # Step 4: Normalize
        normalized = self.normalize(corrected, disk_mask)
        intermediates['normalized'] = normalized.copy()

        # Step 5: Denoise
        denoised = self.denoise(normalized, sigma=1.0)
        intermediates['denoised'] = denoised.copy()

        # Step 6: CLAHE
        enhanced = self.enhance_contrast(denoised, clip_limit=2.0)
        # Apply disk mask
        enhanced[disk_mask == 0] = 0
        intermediates['enhanced'] = enhanced.copy()

        # Step 7: Inversion (for Frangi — filaments become bright ridges)
        inverted = self.invert(enhanced)
        inverted[disk_mask == 0] = 0
        intermediates['inverted'] = inverted.copy()

        # The final preprocessed image for the U-Net is the enhanced version
        # (NOT inverted — the network learns from actual contrast)
        intermediates['preprocessed'] = enhanced.copy()

        if return_intermediates:
            return intermediates
        return {'preprocessed': enhanced, 'inverted': inverted,
                'disk_mask': disk_mask, 'disk_params': (cx, cy, radius)}

    def preprocess_for_model(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess and resize for model input.
        Returns float32 normalized to [0, 1].
        """
        result = self.preprocess(image, return_intermediates=False)
        preprocessed = result['preprocessed']
        resized = self.resize(preprocessed, self.target_size)
        return resized.astype(np.float32) / 255.0

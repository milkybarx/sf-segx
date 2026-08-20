"""
Solar Filament Super-Resolution Dataset & Degradation Pipeline
==============================================================
Extracts high-resolution solar filament patches and generates physics-informed
synthetic low-resolution degraded counterparts for supervised SR training.
"""

import os
import glob
import random
from typing import List, Tuple, Optional
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def apply_solar_degradation(hr_patch: np.ndarray, scale: int = 2,
                            blur_sigma_range: Tuple[float, float] = (0.4, 1.0),
                            noise_std_range: Tuple[float, float] = (0.002, 0.015),
                            jpeg_q_range: Tuple[int, int] = (80, 95)) -> np.ndarray:
    """
    Apply a controlled multi-stage synthetic degradation to an HR patch.
    
    Pipeline:
    1. Atmospheric seeing blur (Gaussian blur)
    2. Spatial downsampling (scale factor 2x or 4x)
    3. Solar sensor Poisson/Gaussian noise
    4. Data transmission compression artifacts (JPEG encoding)
    """
    h, w = hr_patch.shape[:2]
    img = hr_patch.astype(np.float32) / 255.0

    # 1. Atmospheric seeing blur
    sigma = random.uniform(*blur_sigma_range)
    ksize = int(2 * math_ceil(2 * sigma) + 1)
    if ksize % 2 == 0:
        ksize += 1
    if ksize > 1:
        img = cv2.GaussianBlur(img, (ksize, ksize), sigma)

    # 2. Downsample to LR
    lr_w = max(1, w // scale)
    lr_h = max(1, h // scale)
    interp = random.choice([cv2.INTER_CUBIC, cv2.INTER_AREA])
    lr_img = cv2.resize(img, (lr_w, lr_h), interpolation=interp)

    # 3. Add sensor noise
    noise_std = random.uniform(*noise_std_range)
    noise = np.random.normal(0, noise_std, lr_img.shape).astype(np.float32)
    lr_img = np.clip(lr_img + noise, 0.0, 1.0)

    # 4. JPEG compression simulation
    q = random.randint(*jpeg_q_range)
    lr_uint8 = (lr_img * 255.0).astype(np.uint8)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), q]
    _, encimg = cv2.imencode('.jpg', lr_uint8, encode_param)
    lr_decompressed = cv2.imdecode(encimg, cv2.IMREAD_GRAYSCALE if hr_patch.ndim == 2 else cv2.IMREAD_COLOR)

    lr_final = lr_decompressed.astype(np.float32) / 255.0
    return lr_final


def math_ceil(x: float) -> int:
    import math
    return math.ceil(x)


class SolarFilamentSRDataset(Dataset):
    """
    Dataset of solar filament crops and synthetic degraded pairs.
    """
    def __init__(self, image_dir: str, mask_dir: str, patch_size: int = 96,
                 scale_factor: int = 2, num_patches_per_image: int = 20,
                 augment: bool = True, seed: int = 42):
        super().__init__()
        self.patch_size = patch_size
        self.scale_factor = scale_factor
        self.augment = augment

        image_paths = sorted(glob.glob(os.path.join(image_dir, "*.jpeg")) +
                             glob.glob(os.path.join(image_dir, "*.jpg")) +
                             glob.glob(os.path.join(image_dir, "*.png")))
        
        self.patches: List[np.ndarray] = []
        random.seed(seed)
        np.random.seed(seed)

        for img_path in image_paths:
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            mask_path = os.path.join(mask_dir, f"{base_name}.png")

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) if os.path.exists(mask_path) else None

            # Extract filament-centric patches and disk patches
            extracted = self._extract_patches(img, mask, num_patches_per_image)
            self.patches.extend(extracted)

    def _extract_patches(self, img: np.ndarray, mask: Optional[np.ndarray],
                         num_patches: int) -> List[np.ndarray]:
        h, w = img.shape
        patches = []
        
        # Filament coordinates if mask exists
        filament_coords = []
        if mask is not None:
            ys, xs = np.where(mask > 0)
            if len(ys) > 0:
                filament_coords = list(zip(ys, xs))

        ps = self.patch_size
        half_ps = ps // 2

        # 70% filament-centered patches, 30% random disk patches
        num_fil = int(num_patches * 0.7)
        num_bg = num_patches - num_fil

        if filament_coords and num_fil > 0:
            chosen = random.sample(filament_coords, min(num_fil, len(filament_coords)))
            for cy, cx in chosen:
                y0 = max(0, min(h - ps, cy - half_ps))
                x0 = max(0, min(w - ps, cx - half_ps))
                patch = img[y0:y0 + ps, x0:x0 + ps]
                if patch.shape == (ps, ps):
                    patches.append(patch)

        for _ in range(num_bg):
            y0 = random.randint(0, max(0, h - ps))
            x0 = random.randint(0, max(0, w - ps))
            patch = img[y0:y0 + ps, x0:x0 + ps]
            # Avoid off-disk pure black regions
            if patch.shape == (ps, ps) and patch.mean() > 20:
                patches.append(patch)

        return patches

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        hr_patch = self.patches[idx].copy()

        # Data augmentation
        if self.augment:
            if random.random() > 0.5:
                hr_patch = np.fliplr(hr_patch).copy()
            if random.random() > 0.5:
                hr_patch = np.flipud(hr_patch).copy()
            k = random.choice([0, 1, 2, 3])
            if k > 0:
                hr_patch = np.rot90(hr_patch, k).copy()

        # Apply synthetic degradation
        lr_patch = apply_solar_degradation(hr_patch, scale=self.scale_factor)

        # Convert to float tensors [C, H, W] in range [0, 1]
        hr_tensor = torch.from_numpy(hr_patch.astype(np.float32) / 255.0).unsqueeze(0)
        lr_tensor = torch.from_numpy(lr_patch.astype(np.float32)).unsqueeze(0)

        return lr_tensor, hr_tensor
